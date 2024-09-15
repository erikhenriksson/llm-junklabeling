import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class CustomTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context_left = item["context_left"]
        target_text = item["target_text"]
        context_right = item["context_right"]
        label = item["label"]

        # Construct the input sequence
        input_sequence = (
            "<s> " + context_left.strip() + " </s> "
            "<s> " + target_text.strip() + " </s> "
            "<s> " + context_right.strip() + " </s>"
        )

        # Tokenize the input sequence (without padding)
        encoding = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,  # We manually added special tokens
        )

        input_ids = encoding["input_ids"].squeeze(0).tolist()
        attention_mask = encoding["attention_mask"].squeeze(0).tolist()

        # Create the target mask
        # Tokenize context_left and target_text separately to find their lengths
        context_left_encoding = self.tokenizer(
            "<s> " + context_left.strip() + " </s>",
            return_tensors="pt",
            add_special_tokens=False,
        )
        target_text_encoding = self.tokenizer(
            "<s> " + target_text.strip() + " </s>",
            return_tensors="pt",
            add_special_tokens=False,
        )

        len_context_left = context_left_encoding["input_ids"].size(1)
        len_target_text = target_text_encoding["input_ids"].size(1)

        target_start = len_context_left
        target_end = target_start + len_target_text

        target_mask = [0] * len(input_ids)
        for i in range(target_start, target_end):
            if i < len(target_mask):  # Ensure we don't go out of bounds
                target_mask[i] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "labels": label,  # No need to convert to tensor here
        }


from transformers import DataCollatorWithPadding


class DataCollatorWithPaddingAndTargetMask:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    def __call__(self, features):
        # print(features)
        # Extract labels and target_masks
        labels = [feature["labels"] for feature in features]
        target_masks = [feature["target_mask"] for feature in features]

        # Remove labels and target_masks from features to avoid interference
        for feature in features:
            del feature["labels"]
            del feature["target_mask"]

        # Use data_collator to pad input_ids and attention_mask
        batch = self.data_collator(features)

        # Pad target_masks manually
        max_length = batch["input_ids"].shape[1]
        padded_target_masks = torch.zeros(
            (len(target_masks), max_length), dtype=torch.long
        )
        for i, mask in enumerate(target_masks):
            length = mask.size(0)
            padded_target_masks[i, :length] = mask

        batch["target_mask"] = padded_target_masks
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, PreTrainedModel


class CustomRobertaForSequenceClassification(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, target_mask=None, labels=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_length, hidden_size)

        # Apply target mask
        target_mask_expanded = (
            target_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
        target_hidden_states = hidden_states * target_mask_expanded

        # Mean pooling over the target tokens
        sum_hidden = torch.sum(target_hidden_states, dim=1)
        count_nonzero = target_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9).float()
        pooled_output = sum_hidden / count_nonzero

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        output = {"loss": loss, "logits": logits}
        return output


import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class CustomDataCollator:
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        
    def __call__(self, features):
        # Extract labels
        labels = [feature['labels'] for feature in features]
        
        # Extract input_ids, attention_mask, target_mask
        input_ids = [feature['input_ids'] for feature in features]
        attention_masks = [feature['attention_mask'] for feature in features]
        target_masks = [feature['target_mask'] for feature in features]
        
        # Pad input_ids and attention_masks using tokenizer.pad
        batch = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_masks},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Pad target_masks manually
        max_seq_length = batch['input_ids'].shape[1]
        padded_target_masks = torch.zeros((len(target_masks), max_seq_length), dtype=torch.long)
        for i, mask in enumerate(target_masks):
            length = len(mask)
            padded_target_masks[i, :length] = torch.tensor(mask, dtype=torch.long)
        
        batch['target_mask'] = padded_target_masks
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return batch
