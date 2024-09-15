import os
import get_bias_attention_data
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
from datasets import load_from_disk
from dotenv import load_dotenv

import custom

load_dotenv()

testing = os.getenv("TEST_CLASSIFIER") == "1"
create_dataset = os.getenv("CREATE_DATASET") == "1"
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

model = custom.CustomRobertaForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)


# Example usage
train_data, dev_data, test_data = get_bias_attention_data.get_dataset(
    "output/fineweb_annotated_gpt4_multi_2.jsonl"
)

# Create datasets
train_dataset = custom.CustomTextDataset(train_data, tokenizer)
dev_dataset = custom.CustomTextDataset(dev_data, tokenizer)
test_dataset = custom.CustomTextDataset(test_data, tokenizer)

# print an example
print(train_dataset[0])

# Initialize data collator
data_collator = custom.CustomDataCollator(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",  # Directory where model checkpoints and outputs will be saved
    evaluation_strategy="steps",  # Change evaluation strategy to evaluate every N steps
    eval_steps=100,  # Evaluate every 500 steps
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,  # Limit the number of saved checkpoints (optional)
    save_steps=500,  # Save the model every 500 steps
    logging_dir="./logs",  # Directory to store logs (optional)
    remove_unused_columns=False,
)


# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=custom.compute_metrics,
)

trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)
