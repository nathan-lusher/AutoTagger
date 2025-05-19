import os
import subprocess
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ----------- Configuration -------------
VIDEO_PATH = "E:\\DATA\\_New\\_Tagged\\test.mp4"         # Your input video file
WORK_DIR = "clip_temp"              # Temp working directory
FRAMES_DIR = os.path.join(WORK_DIR, "frames")
FPS = 0.01                             # Extract 1 frame per 100 seconds
TOP_N = 10                           # Number of top tags to show
TAGS_FILE = "unique_tags.txt"       # Your list of tags
# --------------------------------------

# Load tag list
with open(TAGS_FILE, "r", encoding="utf-8") as f:
    TAGS = [line.strip() for line in f if line.strip()]

# Make sure temp folders exist
os.makedirs(FRAMES_DIR, exist_ok=True)

# Extract frames using ffmpeg
def extract_frames(video_path, output_folder, fps=1):
    print("Extracting frames...")
    for f in os.listdir(output_folder):  # Clean old frames
        os.remove(os.path.join(output_folder, f))

    command = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(output_folder, "frame_%04d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(command, check=True)
    print(f"Frames extracted to {output_folder}")

# Run CLIP on frames
def classify_frames(folder, tags, top_n=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tag_scores = {tag: 0.0 for tag in tags}
    frame_files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))

    for frame_file in tqdm(frame_files, desc="Classifying"):
        path = os.path.join(folder, frame_file)
        image = Image.open(path).convert("RGB")
        inputs = processor(text=tags, images=image, return_tensors="pt", padding=True, use_fast=True).to(device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        for tag, prob in zip(tags, probs[0]):
            tag_scores[tag] += prob.item()

    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop tags:")
    for tag, score in sorted_tags[:top_n]:
        print(f"{tag}: {score:.2f}")

# Run everything
extract_frames(VIDEO_PATH, FRAMES_DIR, fps=FPS)
classify_frames(FRAMES_DIR, TAGS, top_n=TOP_N)
