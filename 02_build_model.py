import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, TimeDistributed, GlobalMaxPooling1D, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
ROOT_DIR = "E:\\DATA\\_AUTO_TAGGING"
FRAMES_DIR = os.path.join(ROOT_DIR, "frames")
IMAGE_SIZE = (512, 512)
FRAMES_PER_VIDEO = 8
BATCH_SIZE = 8
EPOCHS = 7
# ======================

# --- Load tag index and video labels ---
with open(os.path.join(ROOT_DIR, "tag_index.json"), "r", encoding="utf-8") as f:
    tag_to_index = json.load(f)

with open(os.path.join(ROOT_DIR, "video_labels.json"), "r", encoding="utf-8") as f:
    video_labels = json.load(f)

num_tags = len(tag_to_index)

# --- Build dataset list ---
video_names = list(video_labels.keys())
frame_sets = []
label_vectors = []

for name in video_names:
    frames = [os.path.join(FRAMES_DIR, f"{name}_{i:02d}.jpg")
              for i in range(FRAMES_PER_VIDEO)]
    if all(os.path.exists(fp) for fp in frames):
        frame_sets.append(frames)
        label_vectors.append(video_labels[name])

# --- Split dataset ---
train_frames, val_frames, train_labels, val_labels = train_test_split(
    frame_sets, label_vectors, test_size=0.2, random_state=42
)

# --- Data generator ---


class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, frame_paths, labels, batch_size=BATCH_SIZE):
        self.frame_paths = frame_paths
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.frame_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_frames = self.frame_paths[idx *
                                        self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx *
                                   self.batch_size:(idx + 1) * self.batch_size]

        video_batch = []
        for frame_group in batch_frames:
            frames = [np.array(Image.open(fp).resize(
                IMAGE_SIZE)) / 255.0 for fp in frame_group]
            video_batch.append(frames)

        return np.array(video_batch), np.array(batch_labels)

# --- Build model ---


def build_model(input_shape=(FRAMES_PER_VIDEO, 512, 512, 3), num_tags=100):
    frame_input = Input(shape=input_shape)
    base_cnn = EfficientNetB0(
        include_top=False, pooling='avg', input_shape=input_shape[1:])
    base_cnn.trainable = False

    x = TimeDistributed(base_cnn)(frame_input)
    x = GlobalMaxPooling1D()(x)
    output = Dense(num_tags, activation='sigmoid')(x)

    return Model(inputs=frame_input, outputs=output)


model = build_model(num_tags=num_tags)
model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

# --- Prepare data ---
train_data = VideoDataset(train_frames, train_labels)
val_data = VideoDataset(val_frames, val_labels)

# --- Train ---
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# --- Save model ---
# model.save(os.path.join(ROOT_DIR, "video_tagger_model.h5"))
# model.save(os.path.join(ROOT_DIR, "video_tagger_model.keras"))

print("Saving weights...")
model.save_weights(os.path.join(ROOT_DIR, "video_tagger_model.weights.h5"))
print("✅ Model weights saved.")

# print("Saving model...")
# tf.keras.saving.save_model(model, os.path.join(
#     ROOT_DIR, "video_tagger_model.keras"), save_format='keras')
# print("✅ Model training complete and saved.")
