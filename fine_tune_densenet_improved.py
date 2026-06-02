import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# ===============================
# CONFIGURATION
# ===============================
DATA_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE2 = 15  # fine-tuning longer
UNFREEZE_LAST_LAYERS = 120
BEST_MODEL_PATH = "se_cbam_densenet_best_final.keras"
LATEST_MODEL_PATH = "se_cbam_densenet_latest_final.keras"
LOG_CSV = "training_log_improved.csv"
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ===============================
# LOAD DATASET
# ===============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("✅ Classes:", class_names)

# ===============================
# STRONGER AUGMENTATION
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.35),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.25),
    layers.RandomTranslation(0.1, 0.1),
], name="data_aug")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = val_ds.prefetch(AUTOTUNE)

# ===============================
# BALANCED SAMPLING + CLASS WEIGHTS
# ===============================
y_train = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(1000)], axis=0)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("✅ Class Weights:", class_weights)

# ✅ FIXED balanced sampling (using tf.reduce_any)
from tensorflow.data.experimental import sample_from_datasets

subsets = []
for i in range(num_classes):
    class_subset = train_ds.filter(
        lambda x, y, class_id=i: tf.reduce_any(tf.equal(y, class_id))
    )
    subsets.append(class_subset)

train_ds = sample_from_datasets(subsets, weights=[1/num_classes]*num_classes).prefetch(AUTOTUNE)

# ===============================
# LOAD EXISTING MODEL (RESUME)
# ===============================
if os.path.exists(LATEST_MODEL_PATH):
    print(f"🔁 Resuming from {LATEST_MODEL_PATH}")
    model = tf.keras.models.load_model(LATEST_MODEL_PATH, safe_mode=False)
else:
    raise ValueError("❌ No existing model found. Please train Stage 1 first!")

# ===============================
# UNFREEZE TOP LAYERS FOR FINE-TUNING
# ===============================
print("\n🔧 Fine-tuning top DenseNet layers")
base = model.get_layer("backbone_densenet121")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST_LAYERS]:
    layer.trainable = False

# ===============================
# COSINE DECAY LEARNING RATE
# ===============================
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
# ===============================
# COSINE DECAY LEARNING RATE (Safe Version)
# ===============================
try:
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    if steps_per_epoch <= 0 or steps_per_epoch == np.inf:
        raise ValueError
except Exception:
    # Fallback estimate: assume roughly 80% of data is for training
    num_images = sum(len(files) for _, _, files in os.walk(DATA_DIR))
    est_train = int(num_images * 0.8)
    steps_per_epoch = max(1, est_train // BATCH_SIZE)
    print(f"⚙️ Estimated steps_per_epoch: {steps_per_epoch}")

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch * EPOCHS_STAGE2,
    alpha=1e-6
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ModelCheckpoint(LATEST_MODEL_PATH, monitor='val_accuracy', save_best_only=False, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    CSVLogger(LOG_CSV, append=True)
]

# ===============================
# FINE-TUNE (CONTINUE TRAINING)
# ===============================
print("\n🚀 Continuing fine-tuning (Stage 2)")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# EVALUATION
# ===============================
print("\n🔍 Evaluating best saved model...")
best = tf.keras.models.load_model(BEST_MODEL_PATH, safe_mode=False)
val_loss, val_acc = best.evaluate(val_ds, verbose=1)
print(f"✅ Validation Accuracy: {val_acc*100:.2f}% | Loss: {val_loss:.4f}")

y_true = np.concatenate([y.numpy() for _, y in val_ds.unbatch().batch(1000)], axis=0)
y_pred_probs = best.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
