import os
import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Config ===
ROOT_DIR = 'D:\\DATA\\NATHAN\\Code\\Tagging'
VIDEO_PATH = os.path.join(
    ROOT_DIR, 'E:\\DATA\\_New\\_Tagged\\test.mp4')
FRAMES_DIR = os.path.join(ROOT_DIR, 'frames')
MODEL_PATH = os.path.join(ROOT_DIR, 'model', 'model-resnet_custom_v3.h5')
TAGS_PATH = os.path.join(ROOT_DIR, 'model', 'tags.txt')
FRAME_RATE = 0.01
TOP_N = 100
THRESHOLD = 0.8
# ==============

# Create frames directory if needed
os.makedirs(FRAMES_DIR, exist_ok=True)

# Step 1: Extract frames using ffmpeg


def extract_frames(video_path, output_folder, fps):
    print(f"Extracting frames from {video_path} ...")
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))  # clean old frames

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps},scale=512:512',
        os.path.join(output_folder, 'frame_%04d.jpg'),
        '-hide_banner', '-loglevel', 'error'
    ]
    subprocess.run(command, check=True)
    print("Frames extracted.")

# Step 2: Load model and tags


def load_model_and_tags(model_path, tags_path):
    model = tf.keras.models.load_model(model_path)
    with open(tags_path, 'r', encoding='utf-8') as f:
        tag_names = [line.strip() for line in f]
    return model, tag_names

# Step 3: Predict tags for all frames


def predict_tags(model, tag_names, frame_folder, threshold=0.3, boost=0.05):
    num_tags = len(tag_names)
    max_scores = np.zeros(num_tags)
    counts = np.zeros(num_tags)
    frame_files = sorted(f for f in os.listdir(
        frame_folder) if f.endswith('.jpg'))

    for file in tqdm(frame_files, desc="Tagging frames"):
        image_path = os.path.join(frame_folder, file)
        image = Image.open(image_path)  # .convert('RGB').resize((512, 512))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]

        # Update max scores
        max_scores = np.maximum(max_scores, predictions)

        # Count confident appearances
        counts += predictions >= threshold

    # Final combined score
    combined_scores = max_scores + boost * counts
    return combined_scores


# Step 4: Show top tags
def show_top_tags(tag_scores, tag_names, threshold, top_n):
    top_indices = np.argsort(tag_scores)[::-1]
    print(f"\nTop tags (threshold > {threshold}):")
    count = 0
    for i in top_indices:
        if tag_scores[i] >= threshold:
            print(f"- {tag_names[i]}: {tag_scores[i]:.4f}")
            count += 1
        if count >= top_n:
            break


# === Run ===
extract_frames(VIDEO_PATH, FRAMES_DIR, FRAME_RATE)
model, tag_names = load_model_and_tags(MODEL_PATH, TAGS_PATH)
tag_scores = predict_tags(model, tag_names, FRAMES_DIR)
show_top_tags(tag_scores, tag_names, THRESHOLD, TOP_N)
