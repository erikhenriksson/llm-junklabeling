import json
import os

os.environ["HF_HOME"] = ".hf/hf_home"

from itertools import islice

from datasets import load_dataset
from openai import OpenAI

import prompts

access_token = os.getenv("OPENAI_ACCESS_TOKEN", "")
testing = os.getenv("TEST_OPENAI") == "1"
client = OpenAI(api_key=access_token)
output_file = "output.jsonl"


def generate_response(content):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompts.MESSAGE.format(content)},
        ],
        temperature=0.0,
    )

    model_output = completion.choices[0].message.content

    # Split the output into lines
    model_output = model_output.split("\n")

    # Remove Line number prefixes
    try:
        model_output = [line.split(": ", 1)[1].strip() for line in model_output]
    except:
        print("Error!")
        print(model_output)
        print("Skipping this document")
        return False

    doc_lines = len(content.split("\n"))

    # Check that doc_lines matches length of model output. If not, pad with 'Clean'
    if len(model_output) != doc_lines:
        model_output = model_output + ["Clean"] * (doc_lines - len(model_output))

    return model_output


def split_into_chunks_with_lines(lst, chunk_size=20, min_chunk_size=5):
    chunks = []
    for i, item in enumerate(lst, start=1):
        line_item = f"Line {i}: {item}"
        chunks.append(line_item)

    # Split the processed list into chunks
    split_chunks = [
        chunks[i : i + chunk_size] for i in range(0, len(chunks), chunk_size)
    ]

    # Check if the last chunk has fewer than min_chunk_size items
    if len(split_chunks) > 1 and len(split_chunks[-1]) < min_chunk_size:
        # Concatenate the last small chunk to the previous chunk
        split_chunks[-2].extend(split_chunks[-1])
        split_chunks.pop()  # Remove the last chunk as it's now part of the previous one

    # Join lines within each chunk with newlines
    return ["\n".join(chunk) for chunk in split_chunks]


# Load the dataset by streaming
dataset_url = "HuggingFaceFW/fineweb"
dataset = load_dataset(dataset_url, split="train", streaming=True)
data_sample = list(islice(dataset, 20000))

annotated_ids = set()
with open(output_file, "r", encoding="utf-8") as jsonl_file:
    for line in jsonl_file:
        row = json.loads(line)
        if "id" in row:
            annotated_ids.add(row["id"])

# Process each row in the data sample
for row in data_sample:
    if row.get("id") in annotated_ids:
        continue
    chunks = split_into_chunks_with_lines(row["text"].split("\n"))
    text_annotations = []
    skip_row = False

    for chunk in chunks:
        reply = generate_response(chunk)
        if reply:
            text_annotations.append(reply)
        else:
            skip_row = True

    if skip_row:
        continue

    row["llm_junk_annotations"] = [
        item for sublist in text_annotations for item in sublist
    ]

    # Convert the row to JSON and write to the file
    with open(output_file, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
