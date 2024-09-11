import os
import get_data_for_training_encoder
import numpy as np

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
        model_name, num_labels=7, problem_type="multi_label_classification"
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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset[
        "dev"
    ],  # You can use separate train/eval datasets in practice
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Make predictions
predictions = trainer.predict(tokenized_dataset["test"])
print(predictions.predictions)
