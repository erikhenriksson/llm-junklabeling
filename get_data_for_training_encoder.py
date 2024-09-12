import json
from skmultilearn.model_selection import IterativeStratification
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def clean_annotation(annotation):
    # Strip leading/trailing whitespace and remove extra spaces around semicolons
    return ";".join(part.strip() for part in annotation.split(";")).strip()


def map_annotation(annotation, multilabel):
    labels = annotation.lower().split(";")
    # Map labels to other labels
    label_map = {
        "seo spam": "commercial noise",
        "machine-generated": "commercial noise",
        "placeholder": "technical/boilerplate",
    }
    labels_list = list(set([label_map.get(label, label) for label in labels]))

    if not multilabel:
        labels_list_temp = labels_list[:]
        if len(labels_list) > 1:
            if "clean" in labels_list:
                labels_list = [x for x in labels_list if x != "clean"]
            else:
                labels_list = [x for x in labels_list if x == "navigational"]
        if len(labels_list) > 1 or not labels_list:
            print("this should just have one label:", labels_list, labels_list_temp)
            exit()
        return labels_list[0]

    labels = ";".join(labels_list)
    return labels


def read_jsonl_to_list(file_path, multilabel):
    result_list = []
    # Open the JSONL file
    with open(file_path, "r") as f:
        # Read each line in the JSONL file
        for line in f:
            # Parse the JSON
            data = json.loads(line.strip())
            # Extract 'text' and 'llm_junk_annotations', split text by newlines
            text_lines = data.get("text", "").split("\n")
            # Clean each annotation item
            llm_junk_annotations = [
                map_annotation(clean_annotation(annotation), multilabel)
                for annotation in data.get("llm_junk_annotations", [])
            ]
            # Append a tuple to the result list
            result_list.append((text_lines, llm_junk_annotations))
    return result_list


def get_unique_sorted_labels(parsed_data):
    # Create a set to store unique labels
    unique_labels = set()

    # Loop through each tuple in the parsed data
    for _, annotations in parsed_data:
        for annotation in annotations:
            # Split each annotation by semicolons and add each part to the set
            labels = annotation.split(";")
            unique_labels.update(labels)

    # Convert the set to a sorted list
    sorted_labels = sorted(unique_labels)

    return sorted_labels


def create_one_hot_vectors(annotations, unique_labels):
    # Create a list to hold one-hot vectors for each annotation
    one_hot_vectors = []

    # Loop through each annotation
    for annotation in annotations:
        # Initialize a zero array for one-hot encoding with the same length as unique_labels
        one_hot_vector = np.zeros(len(unique_labels), dtype=int)

        # Split the annotation by semicolons into individual labels
        labels = annotation.split(";")

        # Set 1 for the positions where labels are present
        for label in labels:
            if label in unique_labels:
                index = unique_labels.index(label)
                one_hot_vector[index] = 1

        one_hot_vectors.append(one_hot_vector)

    return one_hot_vectors


def add_one_hot_to_parsed_data(parsed_data, unique_labels, multilabel):
    updated_data = []

    for text_lines, annotations in parsed_data:
        # Generate one-hot vectors for the list of annotations
        if multilabel:
            one_hot_vectors = create_one_hot_vectors(annotations, unique_labels)
        else:

            one_hot_vectors = [unique_labels.index(x) for x in annotations]

        # Append the tuple with text_lines, annotations, and one-hot vectors
        updated_data.append((text_lines, annotations, one_hot_vectors))

    return updated_data


def create_context_window_for_documents(documents, window_size):
    """Creates sliding windows for each document, explicitly marking the target line."""
    context_windows = []
    labels = []
    for doc_idx, document in enumerate(documents):
        num_lines = len(document[0])
        for i in range(num_lines):
            start = max(0, i - window_size)
            end = min(num_lines, i + window_size + 1)

            # Get the context window and mark the target line
            window = (
                document[0][start:i]
                + [f"[TARGET_START] {document[0][i]} [TARGET_END]"]
                + document[0][i + 1 : end]
            )

            # Join the window into a single string
            window_text = " ".join(window)

            # Append the window and its label (label for the target line)
            context_windows.append(window_text)
            labels.append(document[2][i])  # Label corresponds to the target line

    return context_windows, labels


def unique_lists_with_named_counts(list_of_lists, unique_sorted_labels):
    # Convert each list to a tuple so it can be hashed
    list_of_tuples = [tuple(lst) for lst in list_of_lists]

    # Use Counter to count the occurrences of each unique tuple
    counts = Counter(list_of_tuples)

    # Convert the tuples back to lists for the output
    named_counts = {}

    for tup, count in counts.items():
        # Extract the class names from the one-hot vector
        active_classes = [
            unique_sorted_labels[i] for i, val in enumerate(tup) if val == 1
        ]
        # Add the count to the dictionary
        named_counts[tuple(active_classes)] = count

    return named_counts


def get_dataset(file_path, multilabel=True):

    # Example usage
    file_path = "output/fineweb_annotated_gpt4.jsonl"

    # Step 1: Parse the JSONL file into a list of tuples
    parsed_data = read_jsonl_to_list(file_path, multilabel)

    # Step 2: Generate the unique sorted labels
    unique_sorted_labels = get_unique_sorted_labels(parsed_data)

    print(unique_sorted_labels)

    # Step 3: Create the one-hot encoded data
    parsed_data_with_one_hot = add_one_hot_to_parsed_data(
        parsed_data, unique_sorted_labels, multilabel
    )

    print(parsed_data_with_one_hot[0])

    # Step 4: Create context windows and labels for the documents
    texts, labels = create_context_window_for_documents(parsed_data_with_one_hot, 1)

    # Convert lists to numpy arrays
    X = np.array(
        texts
    )  # Features (assuming text representation, which may need further preprocessing)
    Y = np.array(labels)  # Multilabel targets

    if not multilabel:
        # First, split the data into train (70%) and a temporary set (30%)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.3, stratify=Y, random_state=42
        )

        # Next, split the temporary set into dev (10%) and test (20%)
        X_dev, X_test, Y_dev, Y_test = train_test_split(
            X_temp, Y_temp, test_size=2 / 3, stratify=Y_temp, random_state=42
        )

        # Now you have your stratified splits:
        # X_train, Y_train (70% of the data)
        # X_dev, Y_dev (10% of the data)
        # X_test, Y_test (20% of the data)

        # Print the sizes to verify
        print(f"Train set size: {len(X_train)}")
        print(f"Dev set size: {len(X_dev)}")
        print(f"Test set size: {len(X_test)}")

    else:
        named_counts = unique_lists_with_named_counts(Y, unique_sorted_labels)
        print("Named Counts:")
        for classes, count in named_counts.items():
            print(f"Classes: {classes}, Count: {count}")
        # Initialize IterativeStratification for the first split
        # We want 70% train and 30% temporary (which will be further split into 20% test and 10% dev)
        splitter = IterativeStratification(
            n_splits=2, order=1, sample_distribution_per_fold=[0.3, 0.7]
        )

        print(X.shape, Y.shape)

        # First split: Train (70%) and Temporary (30%)
        train_idx, temp_idx = next(splitter.split(X, Y))

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_temp, Y_temp = X[temp_idx], Y[temp_idx]

        # Initialize IterativeStratification for the second split
        # We want 20% test and 10% dev from the temporary set
        splitter_temp = IterativeStratification(
            n_splits=2, order=1, sample_distribution_per_fold=[0.3333, 0.6667]
        )
        test_idx, dev_idx = next(splitter_temp.split(X_temp, Y_temp))

        X_test, Y_test = X_temp[test_idx], Y_temp[test_idx]
        X_dev, Y_dev = X_temp[dev_idx], Y_temp[dev_idx]

        # Print shapes to confirm
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Dev set size: {len(X_dev)}")

    return X_train, Y_train, X_test, Y_test, X_dev, Y_dev
