# -----------------------
# 1️⃣ Imports
# -----------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -----------------------
# 2️⃣ Paths and Params
# -----------------------
base_path = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
img_size = 224      # EfficientNet input size
batch_size = 32
num_classes = 4

# -----------------------
# 3️⃣ Data Generators
# -----------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    base_path,
    target_size=(img_size, img_size),
    color_mode='rgb',          # ✅ ensures 3 channels
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_path,
    target_size=(img_size, img_size),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# -----------------------
# 4️⃣ Base Model (EfficientNetB0)
# -----------------------
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(img_size, img_size, 3)
)

# Freeze base layers (for transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# -----------------------
# 5️⃣ Add Custom Layers
# -----------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()

# -----------------------
# 6️⃣ Compile Model
# -----------------------
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 7️⃣ Callbacks
# -----------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
    ModelCheckpoint('best_efficientnet_model.keras', monitor='val_accuracy', save_best_only=True)
]

# -----------------------
# 8️⃣ Train Model
# -----------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# -----------------------
# 9️⃣ Evaluate
# -----------------------
model.load_weights('best_efficientnet_model.keras')
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n🎯 Final Validation Accuracy: {val_acc*100:.2f}%")

# -----------------------
# 🔟 Plot Accuracy and Loss
# -----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')
plt.show()
