from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import subprocess
from collections import Counter
import json

from tqdm import tqdm

# === Configuration ===
INPUT_DIRS = ["/mnt/e/DATA", "/mnt/s/DATA"]
OUTPUT_DIR = os.path.expanduser("~/tf-data/")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
FRAMES_PER_VIDEO = 20
FRAME_SIZE = (512, 512)
MAX_WORKERS = 6
MAX_TAGS_PER_VIDEO = None  # Set to limit tags per video (e.g. 10), or None
MIN_TAG_FREQUENCY = 10     # Filter tags that appear in fewer than this many videos
# ======================

os.makedirs(FRAMES_DIR, exist_ok=True)

# Step 1: Find all MP4 files and extract tags
video_tags = {}
all_tags = set()
tag_counter = Counter()
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

                # Extract tags inside parentheses
                tags = [tag.strip().lower() for tag in match.group(1).split(",")]

                # Extract prefix tags (e.g., "Vintage Classic" → "vintage", "classic")
                filename = os.path.splitext(file)[0]
                if " - " in filename:
                    prefix = filename.split(" - ")[0].strip().lower()
                    prefix_tags = prefix.split()
                    tags.extend(prefix_tags)

                tags = list(set(tags))  # Remove duplicates

                if MAX_TAGS_PER_VIDEO:
                    tags = tags[:MAX_TAGS_PER_VIDEO]

                video_path = os.path.join(root, file)
                video_tags[video_path] = tags
                tag_counter.update(tags)

# Step 2: Filter tags based on frequency
valid_tags = {tag for tag, count in tag_counter.items() if count >= MIN_TAG_FREQUENCY}
sorted_tags = sorted(valid_tags)
tag_to_index = {tag: i for i, tag in enumerate(sorted_tags)}

print(f"🎯 Using {len(sorted_tags)} tags (min frequency: {MIN_TAG_FREQUENCY})")

# Save tag index and tag counts
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "tag_index.json"), "w", encoding="utf-8") as f:
    json.dump(tag_to_index, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "tag_counts.json"), "w", encoding="utf-8") as f:
    json.dump(tag_counter.most_common(), f, indent=2)

with open(os.path.join(OUTPUT_DIR, "filtered_tag_list.json"), "w", encoding="utf-8") as f:
    json.dump(sorted_tags, f, indent=2)

# Step 3: Clean video tags to remove invalid ones
for path in video_tags:
    video_tags[path] = [tag for tag in video_tags[path] if tag in valid_tags]

# Step 4: Process videos and extract frames
video_frame_map = {}
label_vectors = {}
skipped_videos = []

def process_video(video_path, tags):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_template = os.path.join(FRAMES_DIR, f"{video_name}_%02d.jpg")

    # Get duration via ffprobe
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print(f"⚠️ Skipping: Could not get duration for {video_path}")
        skipped_videos.append(video_path)
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

# Step 5: Run processing in parallel
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_video, path, tags) for path, tags in video_tags.items() if tags]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
        pass

# Step 6: Save label vectors
print("Saving video labels...")
with open(os.path.join(OUTPUT_DIR, "video_labels.json"), "w", encoding="utf-8") as f:
    json.dump(label_vectors, f, indent=2)

print(f"✅ Done. Frames and labels saved to: {OUTPUT_DIR}")
if skipped_videos:
    print(f"⚠️ Skipped {len(skipped_videos)} videos due to errors.")
