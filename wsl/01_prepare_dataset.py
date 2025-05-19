from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import subprocess
from collections import defaultdict
import json

from tqdm import tqdm

# === Configuration ===
# ROOT_DIR = "E:\\DATA"
INPUT_DIRS = ["/mnt/e/DATA", "/mnt/s/DATA"]
OUTPUT_DIR = os.path.expanduser("~/tf-data/")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FRAMES_PER_VIDEO = 20
FRAME_SIZE = (512, 512)
MAX_WORKERS = 6
# ======================

os.makedirs(FRAMES_DIR, exist_ok=True)

# Step 1: Find all MP4 files and extract tags
video_tags = {}
all_tags = set()
pattern = re.compile(r"\((.*?)\)")

for input_dir in INPUT_DIRS:
    for root, _, files in tqdm(os.walk(input_dir), desc=f"Reading tags from {input_dir}"):
        if ("_AUTO_TAGGING" in root) or ("_UNTAGGED" in root):
            continue  # skip output directory
        for file in files:
            if file.lower().endswith(".mp4"):
                match = pattern.search(file)
                if not match:
                    continue

                tags = [tag.strip() for tag in match.group(1).split(",")]

                # Extract prefix as a tag (if it exists)
                filename = os.path.splitext(file)[0]
                if " - " in filename:
                    prefix = filename.split(" - ")[0].strip()
                    if prefix:
                        tags.append(prefix)

                video_path = os.path.join(root, file)
                video_tags[video_path] = tags
                all_tags.update(tags)

# Step 2: Build tag index
sorted_tags = sorted(all_tags)
tag_to_index = {tag: i for i, tag in enumerate(sorted_tags)}

print(f"Discovered {len(video_tags)} videos with {len(sorted_tags)} unique tags.")

print("Saving tag index...")

with open(os.path.join(OUTPUT_DIR, "tag_index.json"), "w", encoding="utf-8") as f:
    json.dump(tag_to_index, f, indent=2)

# Step 3: Process each video
video_frame_map = {}
label_vectors = {}

# Step 3: Define video processing function
def process_video(video_path, tags):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_template = os.path.join(FRAMES_DIR, f"{video_name}_%02d.jpg")

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
        return None

    timestamps = [(duration * i) / FRAMES_PER_VIDEO for i in range(FRAMES_PER_VIDEO)]

    for i, ts in enumerate(timestamps):
        output_path = output_template % i
        if os.path.exists(output_path):
            continue

        command = [
            'ffmpeg', '-hwaccel', 'cuda', '-ss', str(ts), '-i', video_path,
            '-frames:v', '1',
            '-vf', f'scale={FRAME_SIZE[0]}:{FRAME_SIZE[1]}',
            output_path,
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(command)

    frame_paths = [output_template % i for i in range(FRAMES_PER_VIDEO)]
    frame_paths = [f for f in frame_paths if os.path.exists(f)]
    video_frame_map[video_name] = frame_paths

    label_vector = [0] * len(sorted_tags)
    for tag in tags:
        if tag in tag_to_index:
            label_vector[tag_to_index[tag]] = 1
    label_vectors[video_name] = label_vector

# Step 4: Run processing in parallel
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_video, path, tags) for path, tags in video_tags.items()]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
        pass

# Step 5: Save tag index and video labels
print('Saving video labels...')

with open(os.path.join(OUTPUT_DIR, "video_labels.json"), "w", encoding="utf-8") as f:
    json.dump(label_vectors, f, indent=2)

print("✅ Done. Frames and labels saved to:", OUTPUT_DIR)
