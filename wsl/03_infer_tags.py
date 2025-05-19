import os
import json
import subprocess
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, TimeDistributed, GlobalMaxPooling1D, Dense  # type: ignore

# === Config ===
# ROOT_DIR = '/mnt/e/DATA/'
MODEL_DIR = os.path.expanduser("~/tf-models/")
DATA_DIR = os.path.expanduser("~/tf-data/")
VIDEO_PATH = '/mnt/e/DATA/_New/_Tagged/test.mp4'
TEMP_FRAMES_DIR = os.path.expanduser('~/tf-temp/frames/')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'video_tagger_model_v2.weights.h5')
TAG_INDEX_PATH = os.path.join(DATA_DIR, 'tag_index.json')
# FRAME_RATE = 0.01
TOP_N = 100
THRESHOLD = 0.01
FRAMES_PER_VIDEO = 20
# ==============

# === Model architecture (must match training) ===


def build_model(input_shape=(FRAMES_PER_VIDEO, 512, 512, 3), num_tags=100):
    frame_input = Input(shape=input_shape)
    base_cnn = EfficientNetB0(
        include_top=False, pooling='avg', input_shape=input_shape[1:])
    base_cnn.trainable = False

    x = TimeDistributed(base_cnn)(frame_input)
    x = GlobalMaxPooling1D()(x)
    output = Dense(num_tags, activation='sigmoid')(x)

    return Model(inputs=frame_input, outputs=output)

# Step 1: Extract frames using ffmpeg


def extract_frames(video_path, output_folder, frames_per_video=20, frame_size=(512, 512)):
    # Get video duration
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print(f"Could not get duration for: {video_path}")
        return

    timestamps = [(duration * i) / frames_per_video for i in range(frames_per_video)]

    for i, ts in enumerate(timestamps):
        output_path = os.path.join(output_folder, f'frame_{i:04d}.jpg')
        subprocess.run([
            'ffmpeg', '-hwaccel', 'cuda', '-ss', str(ts), '-i', video_path,
            '-frames:v', '1',
            '-vf', f'scale={frame_size[0]}:{frame_size[1]}',
            output_path,
            '-hide_banner', '-loglevel', 'error', '-y'
        ])

# Step 2: Load model and tag index


def load_model_and_tags(weights_path, tag_index_path):
    with open(tag_index_path, 'r', encoding='utf-8') as f:
        tag_to_index = json.load(f)
    index_to_tag = {i: tag for tag, i in tag_to_index.items()}
    tag_names = [index_to_tag[i] for i in range(len(index_to_tag))]

    model = build_model(num_tags=len(tag_names))
    model.load_weights(weights_path)
    return model, tag_names

# Step 3: Predict tags using chunks of 8 frames


def predict_tags(model, tag_names, frame_folder, threshold=0.01, boost=0.05):
    num_tags = len(tag_names)
    max_scores = np.zeros(num_tags)
    counts = np.zeros(num_tags)
    frame_files = sorted(f for f in os.listdir(
        frame_folder) if f.endswith('.jpg'))

    chunks = [frame_files[i:i+FRAMES_PER_VIDEO]
              for i in range(0, len(frame_files), FRAMES_PER_VIDEO)]
    
    # chunks = [sorted(frame_files)]

    for chunk in tqdm(chunks, desc="Tagging frame chunks"):
        if len(chunk) < FRAMES_PER_VIDEO:
            print("Skipping")
            continue  # skip incomplete chunk

        batch = []
        for file in chunk:
            image_path = os.path.join(frame_folder, file)
            image = Image.open(image_path).convert('RGB').resize((512, 512))
            img_array = np.array(image) / 255.0
            batch.append(img_array)

        # shape: (1, 8, 512, 512, 3)
        batch = np.expand_dims(np.stack(batch), axis=0)
        predictions = model.predict(batch, verbose=0)[0]

        max_scores = np.maximum(max_scores, predictions)
        counts += predictions >= threshold

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
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
extract_frames(VIDEO_PATH, TEMP_FRAMES_DIR, FRAMES_PER_VIDEO, (512, 512))
model, tag_names = load_model_and_tags(WEIGHTS_PATH, TAG_INDEX_PATH)
tag_scores = predict_tags(model, tag_names, TEMP_FRAMES_DIR)
show_top_tags(tag_scores, tag_names, THRESHOLD, TOP_N)
