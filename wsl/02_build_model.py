import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, GlobalMaxPooling1D, Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIGURATION ===
INPUT_DIR_ROOT = os.path.expanduser("~/tf-data/")
OUTPUT_DIR = os.path.expanduser("~/tf-models/")
FRAMES_DIR = os.path.join(INPUT_DIR_ROOT, "frames")
IMAGE_SIZE = (512, 512)
FRAMES_PER_VIDEO = 20 # Should be 20 (8 for testing)
BATCH_SIZE = 6
EPOCHS = 20 # Should be 20 (10 for testing)
# ======================

print("Running...")

# Enable mixed precision for faster training on RTX cards
set_global_policy('mixed_float16')

# --- Load tag index and video labels ---
with open(os.path.join(INPUT_DIR_ROOT, "tag_index.json"), "r", encoding="utf-8") as f:
    tag_to_index = json.load(f)

with open(os.path.join(INPUT_DIR_ROOT, "video_labels.json"), "r", encoding="utf-8") as f:
    video_labels = json.load(f)

num_tags = len(tag_to_index)

# --- Build dataset list ---
video_names = list(video_labels.keys())
frame_sets = []
label_vectors = []

for name in tqdm(video_names, "Processing frames"):
    frames = [os.path.join(FRAMES_DIR, f"{name}_{i:02d}.jpg") for i in range(FRAMES_PER_VIDEO)]
    if all(os.path.exists(fp) for fp in frames):
        frame_sets.append(frames)
        label_vectors.append(video_labels[name])

# --- Split dataset ---
train_frames, val_frames, train_labels, val_labels = train_test_split(
    frame_sets, label_vectors, test_size=0.2, random_state=42
)

# --- Augmentation setup ---
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

# --- Data generator ---
class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, frame_paths, labels, batch_size=BATCH_SIZE, augment=False):
        self.frame_paths = frame_paths
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.frame_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_frames = self.frame_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        video_batch = []
        for frame_group in batch_frames:
            frames = []
            for fp in frame_group:
                img = Image.open(fp).resize(IMAGE_SIZE)
                img_array = np.array(img)

                if self.augment:
                    img_array = train_datagen.random_transform(img_array)

                frames.append(img_array / 255.0)

            video_batch.append(frames)

        return np.array(video_batch), np.array(batch_labels)

# --- Build model ---
def build_model(input_shape=(FRAMES_PER_VIDEO, 512, 512, 3), num_tags=100):
    frame_input = Input(shape=input_shape)
    base_cnn = EfficientNetB0(include_top=False, pooling='avg', input_shape=input_shape[1:])
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False

    x = TimeDistributed(base_cnn)(frame_input)
    x = Bidirectional(LSTM(128))(x)
    x = Dense(num_tags, activation='sigmoid', dtype='float32')(x)  # Keep float32 output for stability

    return Model(inputs=frame_input, outputs=x)

model = build_model(num_tags=num_tags)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

# --- Prepare data ---
train_data = VideoDataset(train_frames, train_labels, augment=True)
val_data = VideoDataset(val_frames, val_labels, augment=False)

# --- Set file names --
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_filename = f"video_tagger_model_{timestamp}.weights.h5"
checkpoint_path = os.path.join(OUTPUT_DIR, f"video_tagger_checkpoint_{timestamp}.weights.h5")

# --- Callbacks ---

callbacks = [
    EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_auc', mode='max',
                    save_best_only=True, save_weights_only=True)
]

# --- Train ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- Save weights ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving weights to: {weights_filename}")
model.save_weights(os.path.join(OUTPUT_DIR, weights_filename))
print("✅ Model weights saved.")

# --- Save training history ---
history_df = pd.DataFrame(history.history)
history_file = os.path.join(OUTPUT_DIR, f"train_log_{timestamp}.csv")
history_df.to_csv(history_file, index=False)
print(f"📊 Training log saved to: {history_file}")
