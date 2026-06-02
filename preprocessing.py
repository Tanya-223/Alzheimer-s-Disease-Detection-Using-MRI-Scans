# =====================================
# OASIS MRI Preprocessing (Batch-Safe)
# =====================================

import os
import numpy as np
import cv2
from tqdm import tqdm
import gc  # garbage collection

# ==============================
# CONFIGURATION
# ==============================
DATA_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
SAVE_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\OASIS_preprocessed"
IMG_SIZE = 224
BATCH_SIZE = 5000  # number of images per .npy batch
SLICE_RANGE = (100, 160)
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
# BATCH PROCESSOR
# ==============================
def save_batch(X, y, index):
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.save(os.path.join(SAVE_DIR, f"X_batch_{index}.npy"), X)
    np.save(os.path.join(SAVE_DIR, f"y_batch_{index}.npy"), y)
    print(f"💾 Saved batch {index} → {len(X)} images")
    del X, y
    gc.collect()

def process_and_save_batches():
    batch_X, batch_y = [], []
    batch_index = 0

    for label_idx, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        print(f"\n📂 Loading {cls} images from {folder} ...")

        for fname in tqdm(os.listdir(folder)):
            if not fname.endswith(".jpg") or not is_valid_slice(fname):
                continue

            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.astype(np.float32) / 255.0

            batch_X.append(img)
            batch_y.append(label_idx)

            if len(batch_X) >= BATCH_SIZE:
                save_batch(batch_X, batch_y, batch_index)
                batch_X, batch_y = [], []
                batch_index += 1

    # save leftover images
    if batch_X:
        save_batch(batch_X, batch_y, batch_index)

    print("\n✅ All batches processed and saved at:", SAVE_DIR)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    process_and_save_batches()
