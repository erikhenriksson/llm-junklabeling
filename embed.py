import json

# Initialize an empty set to store unique keyphrases
keyphrases_set = set()

# Path to the JSONL file
file_path = "output/fineweb_annotated.jsonl"

# Read the JSONL file and process each line
with open(file_path, "r") as file:
    for line in file:
        # Parse the JSON object from the line
        json_obj = json.loads(line)
        # Extract the list of keyphrases
        keyphrases = json_obj.get("llm_junk_annotations", [])
        # Add keyphrases to the set
        for keyphrase in keyphrases:
            # Split the keyphrase by semicolon and add to the set
            split_keyphrases = keyphrase.split(";")
            # Add each split keyphrase to the set, trimming whitespace
            keyphrases_set.update([phrase.strip() for phrase in split_keyphrases])

# Convert the set to a sorted list
sorted_keyphrases = list(sorted(keyphrases_set))

# FIlter keywords longer than 50 characters
sorted_keyphrases = [
    keyphrase for keyphrase in sorted_keyphrases if len(keyphrase) <= 50
]


# Print or save the sorted keyphrases
for keyphrase in sorted_keyphrases:
    print(keyphrase)
