import os
import re
import subprocess
from collections import defaultdict
import json

from tqdm import tqdm

# === Configuration ===
ROOT_DIR = "E:\\DATA"
VIDEOS_DIR = "E:\\DATA"
OUTPUT_DIR = os.path.join(ROOT_DIR, "_AUTO_TAGGING")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FRAMES_PER_VIDEO = 8
FRAME_SIZE = (512, 512)
# ======================

os.makedirs(FRAMES_DIR, exist_ok=True)

# Step 1: Find all MP4 files and extract tags
video_tags = {}
all_tags = set()
pattern = re.compile(r"\((.*?)\)")

for root, _, files in os.walk(VIDEOS_DIR):
    if ("_AUTO_TAGGING" in root) or ("_UNTAGGED" in root):
        continue  # skip output directory
    for file in files:
        if file.lower().endswith(".mp4"):
            match = pattern.search(file)
            if not match:
                continue
            tags = [tag.strip() for tag in match.group(1).split(",")]
            video_path = os.path.join(root, file)
            video_tags[video_path] = tags
            all_tags.update(tags)

# Step 2: Build tag index
sorted_tags = sorted(all_tags)
tag_to_index = {tag: i for i, tag in enumerate(sorted_tags)}

print(
    f"Discovered {len(video_tags)} videos with {len(sorted_tags)} unique tags.")

# Step 3: Process each video
video_frame_map = {}
label_vectors = {}

counter = 0

for video_path, tags in tqdm(video_tags.items(), desc="Extracting frames"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_template = os.path.join(FRAMES_DIR, f"{video_name}_%02d.jpg")

    counter += 1
    print(f"Processing {counter} of {len(video_tags)} - '{video_name}'...")

    # Step 1: Get duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print(f"Could not get duration for: {video_path}")

    # Step 2: Calculate timestamps
    timestamps = [(duration * i) /
                  FRAMES_PER_VIDEO for i in range(FRAMES_PER_VIDEO)]

    # Step 3: Extract 1 frame at each timestamp
    for i, ts in enumerate(timestamps):
        output_path = output_template % i  # e.g., 'frames/video_00.jpg'

        if os.path.exists(output_path):
            continue

        command = [
            'ffmpeg', '-ss', str(ts), '-i', video_path,
            '-frames:v', '1',
            '-vf', f'scale={FRAME_SIZE[0]}:{FRAME_SIZE[1]}',
            output_path,
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(command)

    frame_paths = [output_template % i for i in range(FRAMES_PER_VIDEO)]
    frame_paths = [f for f in frame_paths if os.path.exists(f)]
    video_frame_map[video_name] = frame_paths

    # Create label vector
    print('Creating label vector...')

    label_vector = [0] * len(sorted_tags)
    for tag in tags:
        if tag in tag_to_index:
            label_vector[tag_to_index[tag]] = 1
    label_vectors[video_name] = label_vector

# Step 4: Save tag index and video labels
print('Saving output...')

with open(os.path.join(OUTPUT_DIR, "tag_index.json"), "w", encoding="utf-8") as f:
    json.dump(tag_to_index, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "video_labels.json"), "w", encoding="utf-8") as f:
    json.dump(label_vectors, f, indent=2)

print("✅ Done. Frames and labels saved to:", OUTPUT_DIR)
