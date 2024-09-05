import requests
import zstandard as zstd
import json
import os
import io

# URL of the .zst file
url = "https://data.hplt-project.org/one/monotext/fi/1.jsonl.zst"

# Output file path
output_file_path = "data/noisy.jsonl"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Create a stream of the zst file and write the first 100 lines
with requests.get(url, stream=True) as response:
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    with open(output_file_path, "w") as output_file:
        lines_written = 0
        for line in text_stream:
            if lines_written >= 100:
                break
            output_file.write(line)
            lines_written += 1

print(f"First {lines_written} lines written to {output_file_path}")
