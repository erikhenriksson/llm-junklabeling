import os

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Load tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["[TARGET_START]", "[TARGET_END]"]}
)

model = XLMRobertaForSequenceClassification.from_pretrained(
    model_name, num_labels=5
)  # Adjust num_labels to match your categories
model.resize_token_embeddings(
    len(tokenizer)
)  # Resize embeddings to accommodate new tokens


def create_context_window_for_documents(documents, window_size):
    """Creates sliding windows for each document, explicitly marking the target line."""
    context_windows = []
    labels = []
    for doc_idx, document in enumerate(documents):
        num_lines = len(document["lines"])
        for i in range(num_lines):
            start = max(0, i - window_size)
            end = min(num_lines, i + window_size + 1)

            # Get the context window and mark the target line
            window = (
                document["lines"][start:i]
                + [f"[TARGET_START] {document['lines'][i]} [TARGET_END]"]
                + document["lines"][i + 1 : end]
            )

            # Join the window into a single string
            window_text = " ".join(window)

            # Append the window and its label (label for the target line)
            context_windows.append(window_text)
            labels.append(document["labels"][i])  # Label corresponds to the target line

    return context_windows, labels


# Example documents
documents = [
    {
        "lines": [
            "This is the first line of document 1.",
            "Here is some useful information.",
            "Here comes junk content in document 1.",
            "Another important point in document 1.",
            "Finally, the last line of document 1.",
        ],
        "labels": [0, 4, 3, 4, 2],
    },
    {
        "lines": [
            "Document 2 starts here.",
            "Another line in document 2.",
            "Some junk in document 2.",
            "End of document 2.",
        ],
        "labels": [0, 4, 1, 2],
    },
    {
        "lines": [
            "Document 3 starts here.",
            "This is a normal line in document 3.",
            "Final line of document 3.",
        ],
        "labels": [0, 4, 2],
    },
]

# Create context windows and labels for the documents
window_size = 2
context_windows, labels = create_context_window_for_documents(documents, window_size)

# Prepare dataset
data = {"text": context_windows, "label": labels}
dataset = Dataset.from_dict(data)

print(data)
exit()


# Tokenize the context windows
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # You can use separate train/eval datasets in practice
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Make predictions
predictions = trainer.predict(tokenized_dataset)
print(predictions.predictions)
