import os

os.environ["HF_HOME"] = ".hf/hf_home"

import sys


import json
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

remove_words = [
    "information",
    "content",
    "language",
    "rhetoric",
    "statement",
    "reference",
    "mention",
    "description",
    "text",
    "notice",
    "story",
    "commentary",
    "discussion",
]


testing = os.getenv("TEST")

if not testing:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def embed(text):
    if not testing:
        embedding = model.encode([text])["dense_vecs"]

        embedding_str = " ".join([f"{x:.16f}" for x in embedding.flatten()])
        return embedding_str
    else:
        return "1.000 1.000 1.000 1.000"


keyphrases_counter = Counter()
keyphrases_unique_counter = Counter()
keyphrases_coll = {}
file_path = "output/fineweb_annotated_4.jsonl"

# Read the JSONL file and process each line
with open(file_path, "r") as file:
    for line in file:
        # Parse the JSON object from the line
        json_obj = json.loads(line)
        lines = json_obj.get("text", "").split("\n")
        # Extract the list of keyphrases
        keyphrases = json_obj.get("llm_junk_annotations", [])
        # Process each keyphrase
        for k_i, keyphrase in enumerate(keyphrases):
            # Split the keyphrase by semicolon
            cleaned_keyphrase = (
                keyphrase.strip().lower().replace(".", "").replace("\n", "")
            )
            # Update the Counter with each split keyphrase, trimming whitespace
            keyphrases_counter.update(cleaned_keyphrase)

            split_keyphrases = [x.strip() for x in cleaned_keyphrase.split(";")]

            filtered_keyphrases = [
                " ".join(word for word in phrase.split() if word not in remove_words)
                for phrase in split_keyphrases
            ]

            # Update the unique keyphrase Counter
            keyphrases_unique_counter.update(set(filtered_keyphrases))

            line_with_context = (
                (lines[k_i - 2] if k_i >= 2 else "")
                + (lines[k_i - 1] if k_i >= 1 else "")
                + f"[--> {lines[k_i]} <--]"
                + (lines[k_i + 1] if k_i <= len(keyphrases) - 2 else "")
                + (lines[k_i + 2] if k_i <= len(keyphrases) - 3 else "")
            )

            # line_with_context = lines[k_i]

            # Add to collecion
            if cleaned_keyphrase in keyphrases_coll:
                keyphrases_coll[cleaned_keyphrase].append(line_with_context)
            else:
                keyphrases_coll[cleaned_keyphrase] = [line_with_context]

# Convert the Counter to a sorted list of (keyphrase, count) tuples
sorted_keyphrases = sorted(
    keyphrases_counter.items(), key=lambda item: item[1], reverse=True
)

# Filter keywords longer than 50 characters and format the output
filtered_keyphrases = [
    (keyphrase, count) for keyphrase, count in sorted_keyphrases if len(keyphrase) <= 50
]

# Print or save the sorted and filtered keyphrases with their counts
for keyphrase, count in filtered_keyphrases:
    # print("############################################")
    print(f"[{keyphrase}]: {count}")
    # print("Examples:")
    # print examples of this keyphras
    # for ex in keyphrases_coll[keyphrase][:5]:
    #    print("------")
    #    print(ex)


# Print the unique keyphrases and their counts
print("############################################")
print("Unique Keyphrases:")
for keyphrase, count in keyphrases_unique_counter.items():
    print(f"[{keyphrase}]: {count}")

    embedding = embed(keyphrase)

    # Save embedding to csv file
    with open("output/embeddings.tsv", "a") as f:
        f.write(f"{keyphrase}\t{count}\t{embedding}\n")
