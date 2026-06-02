# oasis_data_generator.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
IMG_SIZE = 224
BATCH_SIZE = 32

# Automatically split data into train/val/test (80/10/10)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,     # 80% train, 20% for val+test
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    color_mode='rgb',          # ✅ ensures 3 channels
    subset='training',
    shuffle=True
)

val_gen = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    color_mode='rgb',
    subset='validation',
    shuffle=True
)
