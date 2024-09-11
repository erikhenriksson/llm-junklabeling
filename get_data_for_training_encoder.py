import json
from skmultilearn.model_selection import IterativeStratification
import numpy as np


def clean_annotation(annotation):
    # Strip leading/trailing whitespace and remove extra spaces around semicolons
    return ";".join(part.strip() for part in annotation.split(";")).strip()


def read_jsonl_to_list(file_path):
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
                clean_annotation(annotation)
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


def add_one_hot_to_parsed_data(parsed_data, unique_labels):
    updated_data = []

    for text_lines, annotations in parsed_data:
        # Generate one-hot vectors for the list of annotations
        one_hot_vectors = create_one_hot_vectors(annotations, unique_labels)

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

def get_dataset(file_path):

    # Example usage
    file_path = "output/fineweb_annotated_gpt4.jsonl"

    # Step 1: Parse the JSONL file into a list of tuples
    parsed_data = read_jsonl_to_list(file_path)

    # Step 2: Generate the unique sorted labels
    unique_sorted_labels = get_unique_sorted_labels(parsed_data)

    print(unique_sorted_labels)

    # Step 3: Create the one-hot encoded data
    parsed_data_with_one_hot = add_one_hot_to_parsed_data(parsed_data, unique_sorted_labels)

    # print(parsed_data_with_one_hot[0])

    # Step 4: Create context windows and labels for the documents
    texts, labels = create_context_window_for_documents(parsed_data_with_one_hot, 1)

    # Convert lists to numpy arrays
    X = np.array(
        texts
    )  # Features (assuming text representation, which may need further preprocessing)
    Y = np.array(labels)  # Multilabel targets

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
