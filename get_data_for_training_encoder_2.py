import json
from skmultilearn.model_selection import IterativeStratification
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import random

random.seed(42)


def clean_annotation(annotation):
    # Strip leading/trailing whitespace and remove extra spaces around semicolons
    return ";".join(part.strip() for part in annotation.split(";")).strip()


def map_annotation(annotation):
    labels = annotation.lower().split(";")
    if "clean" in labels:
        labels = ["clean"]
    else:
        labels = ["junk"]
    # else:
    #    if len(labels) > 1:
    #        labels = ["other junk"]

    label = labels[0]

    if label in ["garbled", "code"]:
        label = "other junk"

    return label


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
                map_annotation(clean_annotation(annotation))
                for annotation in data.get("llm_junk_annotations", [])
            ]

            # Append a tuple to the result list
            result_list.append((text_lines, llm_junk_annotations))
    return result_list


def get_unique_sorted_labels(parsed_data):
    # Create a set to store unique labels
    unique_labels = set()

    # Count the occurrences of each label
    label_counts = Counter()

    # Loop through each tuple in the parsed data
    for _, annotations in parsed_data:

        for annotation in annotations:

            unique_labels.add(annotation)
            label_counts[annotation] += 1

    # Convert the set to a sorted list
    sorted_labels = sorted(unique_labels)

    return sorted_labels, label_counts


def add_one_hot_to_parsed_data(parsed_data, unique_labels):
    updated_data = []

    for text_lines, annotations in parsed_data:
        one_hot_vectors = [unique_labels.index(x) for x in annotations]

        # Append the tuple with text_lines, annotations, and one-hot vectors
        updated_data.append((text_lines, annotations, one_hot_vectors))

    return updated_data


def create_context_window(document, window_size):
    """Creates sliding windows for each document, explicitly marking the target line."""
    context_windows = []

    num_lines = len(document)
    for i in range(num_lines):
        if not window_size:
            window = [document[i]]
        else:
            start = max(0, i - window_size)
            end = min(num_lines, i + window_size + 1)

            # Get the context window and mark the target line
            window = (
                document[start:i]
                + [f"[TARGET_START] {document[i]} [TARGET_END]"]
                + document[i + 1 : end]
            )

        # Join the window into a single string
        window_text = " ".join(window)

        # Append the window and its label (label for the target line)
        context_windows.append(window_text)
    return context_windows


def downsample_class(X, Y, M, downsample_ratio=0.25, random_seed=42):

    np.random.seed(random_seed)  # For reproducibility

    # Get unique classes
    unique_classes = np.unique(Y)

    # List to store indices for all classes
    selected_indices = []

    for class_label in unique_classes:
        class_indices = np.where(Y == class_label)[0]

        if class_label == M:
            # Calculate the desired number of class M samples based on the downsample ratio
            total_samples = len(Y)
            desired_class_M_count = int(downsample_ratio * total_samples)

            # Randomly sample from class M
            downsampled_class_M_indices = np.random.choice(
                class_indices, size=desired_class_M_count, replace=False
            )
            selected_indices.extend(downsampled_class_M_indices)
        else:
            # Keep all samples for other classes
            selected_indices.extend(class_indices)

    # Shuffle the combined indices
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)

    # Downsample the feature matrix and labels
    X_downsampled = X[selected_indices]
    Y_downsampled = Y[selected_indices]

    return X_downsampled, Y_downsampled


def get_dataset(file_path):

    # Step 1: Parse the JSONL file into a list of tuples
    parsed_data = read_jsonl_to_list(file_path)

    # Step 2: Generate the unique sorted labels
    unique_sorted_labels, counts = get_unique_sorted_labels(parsed_data)

    print(unique_sorted_labels)
    print(counts)

    print(parsed_data[0])

    labels = []
    texts = []

    for doc in parsed_data:
        labels += [unique_sorted_labels.index(x) for x in doc[1]]
        texts += create_context_window(doc[0], window_size=0)

    print("context windows ready")

    # Convert lists to numpy arrays
    X = np.array(
        texts
    )  # Features (assuming text representation, which may need further preprocessing)
    Y = np.array(labels)  # Multilabel targets

    X, Y = downsample_class(X, Y, 0)

    # First, split the data into train (70%) and a temporary set (30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42
    )

    # Next, split the temporary set into dev (10%) and test (20%)
    X_dev, X_test, Y_dev, Y_test = train_test_split(
        X_temp, Y_temp, test_size=2 / 3, stratify=Y_temp, random_state=42
    )

    # Print the sizes to verify
    print(f"Train set size: {len(X_train)}")
    print(f"Dev set size: {len(X_dev)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, Y_train, X_test, Y_test, X_dev, Y_dev
