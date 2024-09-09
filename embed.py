import json
from collections import Counter

# Initialize a Counter to store keyphrases and their counts
keyphrases_counter = Counter()
keyphrases_coll = {}
# Path to the JSONL file
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
            split_keyphrases = keyphrase.strip().lower().replace(".", "")
            # Update the Counter with each split keyphrase, trimming whitespace
            keyphrases_counter.update(
                phrase.strip().replace("\n", "")
                for phrase in [split_keyphrases]
                if phrase
            )

            # Add to collecion
            if split_keyphrases in keyphrases_coll:
                keyphrases_coll[split_keyphrases].append(lines[k_i])
            else:
                keyphrases_coll[split_keyphrases] = [lines[k_i]]

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
    print(f"[{keyphrase}]: {count}")
    print("Examples:")
    # print examples of this keyphras
    for ex in keyphrases_coll[keyphrase][:5]:
        print("------")
        print(ex)
