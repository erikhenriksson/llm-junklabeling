import torch
import json
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

# The path of your model after cloning it
model_dir = "Marqo/dunzhang-stella_en_400M_v5"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def get_embedding(text):
    with torch.no_grad():
        # Tokenize and prepare input data
        input_data = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]

        # Get the model's last hidden states
        last_hidden_state = model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        # Calculate the embedding
        embedding = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = normalize(embedding.cpu().numpy())

    print(embedding.flatten())
    return embedding.flatten()  # Flatten to make it easy to save to CSV


# Load the JSON file
with open("unique_label_counts.json", "r") as f:
    data = json.load(f)

# Prepare data for CSV
records = []
for text, count in data.items():
    embedding = get_embedding(text)
    records.append(
        {
            "text": text,
            "count": count,
            "embedding": embedding.tolist(),  # Convert to list to store in CSV
        }
    )

# Create DataFrame and save to CSV
df = pd.DataFrame(records)
df.to_csv("embeddings_output.csv", index=False)

print("Embeddings saved to embeddings_output.csv")
