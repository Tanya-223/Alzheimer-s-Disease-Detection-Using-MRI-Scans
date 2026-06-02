# =============================================
# OASIS 3D MRI Data Generator (Volumetric CNN)
# =============================================

import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
import gc

# ==============================
# CONFIGURATION
# ==============================
DATA_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
SAVE_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\OASIS_preprocessed_3D"
IMG_SIZE = 128        # resize to smaller size for 3D efficiency
SLICE_RANGE = (100, 160)
SLICES_PER_VOLUME = 16  # number of slices per 3D input
CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# SLICE FILTER
# ==============================
def is_valid_slice(filename):
    try:
        slice_num = int(filename.split('_')[-1].split('.')[0])
        return SLICE_RANGE[0] <= slice_num <= SLICE_RANGE[1]
    except:
        return False

# ==============================
# STEP 1: BUILD 3D VOLUMES
# ==============================
def build_3d_volumes():
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        class_dir = os.path.join(SAVE_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)

        print(f"\n📂 Processing {cls} slices into 3D volumes...")
        slice_files = sorted([f for f in os.listdir(folder) if is_valid_slice(f) and f.endswith(".jpg")])

        # process in sliding windows of SLICES_PER_VOLUME
        for i in tqdm(range(0, len(slice_files) - SLICES_PER_VOLUME, SLICES_PER_VOLUME)):
            volume = []
            for j in range(SLICES_PER_VOLUME):
                img_path = os.path.join(folder, slice_files[i + j])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                volume.append(img)
            
            if len(volume) == SLICES_PER_VOLUME:
                volume = np.stack(volume, axis=-1)  # shape: (128, 128, 16)
                np.save(os.path.join(class_dir, f"vol_{i // SLICES_PER_VOLUME}.npy"), volume)

            gc.collect()
    print("\n✅ All 3D volumes saved successfully!")

# ==============================
# STEP 2: 3D DATA GENERATOR
# ==============================
class OASIS3DGenerator(Sequence):
    def __init__(self, folder_paths, batch_size=4, augment=False, shuffle=True):
        self.file_paths = folder_paths
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for path in batch_files:
            vol = np.load(path)
            vol = vol.astype(np.float32) / 255.0
            vol = np.expand_dims(vol, axis=-1)  # (128, 128, 16, 1)
            X.append(vol)
            if "Non Demented" in path:
                y.append(0)
            elif "Very mild Dementia" in path:
                y.append(1)
            elif "Mild Dementia" in path:
                y.append(2)
            else:
                y.append(3)

        X, y = np.array(X), np.array(y)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_paths)


# ==============================
# STEP 3: CREATE GENERATORS
# ==============================
def get_3d_generators():
    all_files = []
    for cls in CLASSES:
        folder = os.path.join(SAVE_DIR, cls)
        all_files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npy")]

    np.random.shuffle(all_files)
    train_split = int(0.7 * len(all_files))
    val_split = int(0.85 * len(all_files))

    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]

    train_gen = OASIS3DGenerator(train_files, batch_size=4)
    val_gen = OASIS3DGenerator(val_files, batch_size=4)
    test_gen = OASIS3DGenerator(test_files, batch_size=4)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    build_3d_volumes()
