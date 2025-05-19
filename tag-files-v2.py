import os
import subprocess
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import deepdanbooru as dd
import numpy as np

# ---------- CONFIG ----------
VIDEO_PATH = "E:\\DATA\\_New\\_Tagged\\test.mp4"
WORK_DIR = "clip_temp"
FRAMES_DIR = os.path.join(WORK_DIR, "frames")
FPS = 0.2  # 1 frame every 5 seconds
TOP_N = 5
TAGS_FILE = "unique_tags.txt"
DEEPDANBOORU_MODEL_PATH = "D:\\DATA\\NATHAN\\Code\\Tagging\\DeepDanbooru\\deepdanbooru\\model"
DD_THRESHOLD = 0.3
# ----------------------------

# Create working folders
os.makedirs(FRAMES_DIR, exist_ok=True)

# Load tag list for CLIP
with open(TAGS_FILE, "r", encoding="utf-8") as f:
    CLIP_TAGS = [line.strip() for line in f if line.strip()]

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load DeepDanbooru model
dd_model = dd.project.load_project(DEEPDANBOORU_MODEL_PATH)
dd_tags = dd.project.load_tags_from_project(DEEPDANBOORU_MODEL_PATH)

# Step 1: Extract frames
def extract_frames(video_path, output_folder, fps):
    print("Extracting frames...")
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_folder, "frame_%04d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ], check=True)

# Step 2: Run CLIP on a frame
def classify_clip(image):
    inputs = clip_processor(text=CLIP_TAGS, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return {tag: prob.item() for tag, prob in zip(CLIP_TAGS, probs[0])}

# Step 3: Run DeepDanbooru on a frame
def classify_deepdanbooru(image):
    img_arr = np.array(image.resize((512, 512))).astype(np.float32) / 255
    img_arr = np.expand_dims(img_arr, axis=0)
    preds = dd_model.predict(img_arr)[0]
    tags = {tag: preds[i] for i, tag in enumerate(dd_tags) if preds[i] >= DD_THRESHOLD}
    return tags

# Run both taggers and aggregate results
def run_tagger():
    clip_scores = {tag: 0.0 for tag in CLIP_TAGS}
    dd_scores = {}

    frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))

    for frame_file in tqdm(frame_files, desc="Tagging frames"):
        path = os.path.join(FRAMES_DIR, frame_file)
        image = Image.open(path).convert("RGB")

        # CLIP tags
        clip_result = classify_clip(image)
        for tag, score in clip_result.items():
            clip_scores[tag] += score

        # DeepDanbooru tags
        dd_result = classify_deepdanbooru(image)
        for tag, score in dd_result.items():
            dd_scores[tag] = dd_scores.get(tag, 0) + score

    # Final results
    top_clip = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    top_dd = sorted(dd_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

    print("\n=== CLIP Tags ===")
    for tag, score in top_clip:
        print(f"{tag}: {score:.2f}")

    print("\n=== DeepDanbooru Tags ===")
    for tag, score in top_dd:
        print(f"{tag}: {score:.2f}")

# Run full process
extract_frames(VIDEO_PATH, FRAMES_DIR, FPS)
run_tagger()
