from datasets import load_dataset

# Load the dataset by streaming
dataset_url = "HuggingFaceFW/fineweb"
dataset = load_dataset(dataset_url, split="train", streaming=True)

# Fetch the first 100 rows by using islice to limit the stream
from itertools import islice

# Limit the streaming to the first 100 rows
first_100_rows = list(islice(dataset, 100))

# Print or process the first 100 rows
for row in first_100_rows:
    lines = row["text"].split("\n")
    print("------")
    for line in lines:
        print(line)
