import json
from collections import Counter

# Initialize a Counter to store keyphrases and their counts
keyphrases_counter = Counter()

# Path to the JSONL file
file_path = "output/fineweb_annotated_2.jsonl"

# Read the JSONL file and process each line
with open(file_path, "r") as file:
    for line in file:
        # Parse the JSON object from the line
        json_obj = json.loads(line)
        # Extract the list of keyphrases
        keyphrases = json_obj.get("llm_junk_annotations", [])
        # Process each keyphrase
        for keyphrase in keyphrases:
            # Split the keyphrase by semicolon
            split_keyphrases = keyphrase.split(";")
            # Update the Counter with each split keyphrase, trimming whitespace
            keyphrases_counter.update(phrase.strip() for phrase in split_keyphrases)

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
    print(f"{keyphrase}: {count}")
