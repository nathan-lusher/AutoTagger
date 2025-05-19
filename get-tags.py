import os
import re

# Root folder containing videos
VIDEO_ROOT = "E:\\DATA"

# Regex to extract text in parentheses
pattern = re.compile(r"\((.*?)\)")

all_tags = set()

for dirpath, _, filenames in os.walk(VIDEO_ROOT):
    for filename in filenames:
        if filename.lower().endswith(".mp4"):
            match = pattern.search(filename)
            if match:
                tags_str = match.group(1)
                tags = [tag.strip() for tag in tags_str.split(",")]
                all_tags.update(tags)

# Sort and print the unique tags
sorted_tags = sorted(all_tags)
print("Unique tags found:")
for tag in sorted_tags:
    print(f"- {tag}")

# Optional: save to a file
with open("unique_tags.txt", "w", encoding="utf-8") as f:
    for tag in sorted_tags:
        f.write(f"{tag}\n")
