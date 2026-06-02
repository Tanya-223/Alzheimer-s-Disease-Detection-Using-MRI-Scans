import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from sklearn.metrics import classification_report
import numpy as np
import os

# ===============================
# Dataset Path
# ===============================
data_dir = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"

# ===============================
# Load Dataset (auto-split)
# ===============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ===============================
# Data Augmentation
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
])

# ===============================
# Prefetch for performance
# ===============================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ===============================
# Attention Block (works with Keras tensors)
# ===============================
def attention_block(inputs):
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    avg_pool = layers.Reshape((1, 1, inputs.shape[-1]))(avg_pool)
    dense1 = layers.Dense(inputs.shape[-1] // 8, activation='relu')(avg_pool)
    dense2 = layers.Dense(inputs.shape[-1], activation='sigmoid')(dense1)
    return layers.Multiply()([inputs, dense2])

# ===============================
# Build Attention + DenseNet Model
# ===============================
def build_attention_densenet(input_shape=(224, 224, 3), num_classes=4):
    base_model = applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base layers

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = attention_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

# ===============================
# Compile Model
# ===============================
model = build_attention_densenet(num_classes=len(class_names))
model.compile(optimizer=optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ===============================
# Train Model
# ===============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ===============================
# Evaluation
# ===============================
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
