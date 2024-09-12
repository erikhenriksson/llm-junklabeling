import os
import get_data_for_training_encoder
import numpy as np
from sklearn.metrics import classification_report, f1_score

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

testing = os.getenv("TEST_CLASSIFIER") == "1"
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

if not testing:
    # Load tokenizer and model

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[TARGET_START]", "[TARGET_END]"]}
    )

    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=5, problem_type="multi_label_classification"
    )
    model.resize_token_embeddings(len(tokenizer))

# Example usage
X_train, y_train, X_test, y_test, X_dev, y_dev = (
    get_data_for_training_encoder.get_dataset("output/fineweb_annotated_gpt4.jsonl")
)

X_train = np.array(X_train)
y_train = np.array(y_train).astype(float)
X_test = np.array(X_test)
y_test = np.array(y_test).astype(float)
X_dev = np.array(X_dev)
y_dev = np.array(y_dev).astype(float)

train_dataset = Dataset.from_dict({"text": X_train, "labels": y_train})
dev_dataset = Dataset.from_dict({"text": X_dev, "labels": y_dev})
test_dataset = Dataset.from_dict({"text": X_test, "labels": y_test})

# Combine into DatasetDict
dataset_dict = DatasetDict(
    {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
)


# Tokenize the context windows
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)


# Define the custom evaluation metrics function
def compute_metrics(pred):
    # Get the true labels and the predicted logits or probabilities
    labels = pred.label_ids
    preds = pred.predictions

    # Convert logits or probabilities to binary predictions using a threshold of 0.5
    preds = (preds > 0.5).astype(int)

    # Calculate micro and macro F1 scores
    micro_f1 = f1_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")

    # Generate classification report to inspect per-class metrics
    class_report = classification_report(labels, preds, output_dict=True)

    # You can log or print the classification report
    print("Classification Report:")
    print(classification_report(labels, preds))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "class_report": class_report,  # Include the detailed per-class report
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory where model checkpoints and outputs will be saved
    evaluation_strategy="steps",  # Change evaluation strategy to evaluate every N steps
    eval_steps=250,  # Evaluate every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,  # Limit the number of saved checkpoints (optional)
    save_steps=500,  # Save the model every 500 steps
    logging_dir="./logs",  # Directory to store logs (optional)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    compute_metrics=compute_metrics,  # Add custom evaluation metrics function
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./fine_tuned_model")  # Save the model
tokenizer.save_pretrained("./fine_tuned_model")  # Save the tokenizer

# Evaluate the model
results = trainer.evaluate()
print(results)

# Make predictions
predictions = trainer.predict(tokenized_dataset["test"])
print(predictions.predictions)
