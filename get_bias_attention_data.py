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
    label = annotation.lower()

    if label != "clean":
        label = "junk"

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


def create_context_window(document, label, window_size):

    context_windows = []
    num_lines = len(document)

    for i in range(num_lines):
        start = max(0, i - window_size)
        end = min(num_lines, i + window_size + 1)
        try:
            this_label = label[i]
        except:
            print("Warning: label not found")
            this_label = 0
        window = {
            "context_left": "\n".join(document[start:i]),
            "target_text": document[i],
            "context_right": "\n".join(document[i + 1 : end]),
            "label": this_label,
        }
        context_windows.append(window)
    return context_windows


def downsample_class(X, Y, M, downsample_ratio=0.3, random_seed=42):

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
        # print(doc)

        label = [unique_sorted_labels.index(x) for x in doc[1]]
        labels += label
        texts += create_context_window(doc[0], label, window_size=1)

    print("context windows ready")

    print(texts[0])

    # Extract labels for stratification
    labels = [item["label"] for item in texts]

    # Convert lists to numpy arrays
    X = np.array(
        texts
    )  # Features (assuming text representation, which may need further preprocessing)
    Y = np.array(labels)  # Multilabel targets

    texts, labels = downsample_class(X, Y, 0)

    # Split data into train+dev and test sets (80% train+dev, 20% test)
    train_dev_data, test_data = train_test_split(
        texts, test_size=0.2, stratify=labels, random_state=42
    )

    # Extract labels for the train_dev split
    train_dev_labels = [item["label"] for item in train_dev_data]

    # Split train_dev_data into train and dev sets (75% train, 25% dev of train_dev_data)
    train_data, dev_data = train_test_split(
        train_dev_data, test_size=0.25, stratify=train_dev_labels, random_state=42
    )

    # Now we have train_data (60%), dev_data (20%), test_data (20%)
    print(f"Train size: {len(train_data)}")
    print(f"Dev size: {len(dev_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, dev_data, test_data
