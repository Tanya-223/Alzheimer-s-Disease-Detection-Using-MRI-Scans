import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

# ===============================================================
# CONFIGURATION
# ===============================================================
DATA_PATH = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"   # 🔹 change this to your dataset path
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "attention_densenet_best.keras"
LOG_PATH = "training_log.csv"

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ===============================================================
# DATASET LOADING
# ===============================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ get class names before caching
class_names = train_ds.class_names
print("✅ Classes:", class_names)

# ⚡ Optimize dataset loading for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ===============================================================
# CBAM BLOCK
# ===============================================================
def cbam_block(feature_map, ratio=8):
    channel = feature_map.shape[-1]

    # ----- Channel Attention -----
    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    max_pool = layers.GlobalMaxPooling2D()(feature_map)

    shared_dense_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal')
    shared_dense_two = layers.Dense(channel, activation='sigmoid', kernel_initializer='he_normal')

    avg_dense = shared_dense_two(shared_dense_one(avg_pool))
    max_dense = shared_dense_two(shared_dense_one(max_pool))

    channel_attention = layers.Add()([avg_dense, max_dense])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, channel))(channel_attention)
    x = layers.Multiply()([feature_map, channel_attention])

    # ----- Spatial Attention (⚙️ using Lambda properly) -----
    avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, padding='same',
                                      activation='sigmoid', kernel_initializer='he_normal')(concat)
    x = layers.Multiply()([x, spatial_attention])
    return x

# ===============================================================
# MODEL DEFINITION
# ===============================================================
def build_attention_densenet(input_shape=(128, 128, 3), num_classes=4):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:200]:
        layer.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===============================================================
# MODEL INITIALIZATION
# ===============================================================
model = build_attention_densenet(input_shape=(128, 128, 3), num_classes=len(class_names))
model.summary()

# ===============================================================
# CALLBACKS
# ===============================================================
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
csv_logger = CSVLogger(LOG_PATH, append=True)

# ===============================================================
# TRAINING
# ===============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, csv_logger],
    verbose=1
)

# ===============================================================
# SAVE FINAL MODEL
# ===============================================================
model.save("attention_densenet_final.keras")
print("✅ Training complete! Model saved as 'attention_densenet_final.keras'")

# ===============================================================
# EVALUATION (Accuracy, Precision, Recall, F1)
# ===============================================================
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
