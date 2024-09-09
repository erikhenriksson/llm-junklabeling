import os
import json

os.environ["HF_HOME"] = ".hf/hf_home"

from itertools import islice

from datasets import load_dataset

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import SYSTEM

load_dotenv()


access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")
testing = os.getenv("TEST")
if not testing:
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=access_token,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def generate_responses(windows):

    if testing:
        return ["This is a sample response"] * len(windows)

    messages = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": window},
        ]
        for window in windows
    ]

    # Tokenize the batch of prompts with padding and truncation
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        attn_implementation="flash_attention_2",
        padding=True,
        truncation=True,
    ).to("cuda")

    # Generate responses for the batch
    outputs = model.generate(
        **inputs, do_sample=True, temperature=0.01, max_new_tokens=64
    )

    # Decode the batch of outputs
    batch_outputs = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return batch_outputs


def generate_window(document, target_idx, window_size=2):

    num_lines = len(document)

    # Determine the start and end indices for the context window
    start = max(0, target_idx - window_size)
    end = min(num_lines, target_idx + window_size + 1)

    # Initialize the window with context lines before the target
    window = []

    # Add [DOC_START] if the target line is the first line in the document
    if target_idx == 0:
        window.append("[DOC_START]")

    # Add the lines before the target line
    for i in range(start, target_idx):
        window.append(f"Context Line {i - start + 1}: {document[i]}")

    # Add the target line, marked with [TARGET_START] and [TARGET_END]
    window.append(f"[TARGET_START] {document[target_idx]} [TARGET_END]")

    # Add the lines after the target line
    for i in range(target_idx + 1, end):
        window.append(f"Context Line {i - target_idx}: {document[i]}")

    # Add [DOC_END] if the target line is the last line in the document
    if target_idx == num_lines - 1:
        window.append("[DOC_END]")

    # Join the window list into a single string with each line on a new line
    return "\n".join(window)


def generate_all_windows(document, window_size=2):

    all_windows = []

    # Iterate over all lines in the document
    for target_idx in range(len(document)):
        # Generate the window for the current target line
        window = generate_window(document, target_idx, window_size)
        all_windows.append(window)

    return all_windows


# Load the dataset by streaming
dataset_url = "HuggingFaceFW/fineweb"
output_file = "output/fineweb_annotated_4.jsonl"
dataset = load_dataset(dataset_url, split="train", streaming=True)
data_sample = list(islice(dataset, 1000))

annotated_ids = set()
with open(output_file, "r", encoding="utf-8") as jsonl_file:
    for line in jsonl_file:
        row = json.loads(line)
        if "id" in row:
            annotated_ids.add(row["id"])

# Initialize a batch collection list
batch_size = 4

# Process each row in the data sample
for row in data_sample:
    if row.get("id") in annotated_ids:
        continue
    windows = generate_all_windows(row["text"].split("\n"), 2)
    text_annotations = []
    batch = []
    # Add each generated window to the batch
    for window in windows:
        batch.append(window)

        # If the batch size is reached, process the batch
        if len(batch) == batch_size:
            text_annotations += generate_responses(batch)
            batch = []  # Clear the batch after processing

    # Process any remaining windows that didn't form a complete batch
    if batch:
        text_annotations += generate_responses(batch)

    row["llm_junk_annotations"] = text_annotations

    # Convert the row to JSON and write to the file
    with open(output_file, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
