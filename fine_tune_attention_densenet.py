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
EPOCHS_STAGE1 = 6
EPOCHS_STAGE2 = 10
UNFREEZE_LAST_LAYERS = 80
BEST_MODEL_PATH = "se_cbam_densenet_best_final.keras"
LATEST_MODEL_PATH = "se_cbam_densenet_latest_final.keras"
LOG_CSV = "training_log_final.csv"
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
# DATA AUGMENTATION
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.2),
], name="data_aug")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = val_ds.prefetch(AUTOTUNE)

# ===============================
# CLASS WEIGHTS
# ===============================
y_train = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(1000)], axis=0)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("✅ Class Weights:", class_weights)

# ===============================
# ATTENTION BLOCKS
# ===============================
def se_block(inputs, ratio=8, name="se"):
    ch = inputs.shape[-1]
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(inputs)
    x = layers.Dense(ch // ratio, activation='relu', name=f"{name}_fc1")(x)
    x = layers.Dense(ch, activation='sigmoid', name=f"{name}_fc2")(x)
    x = layers.Reshape((1, 1, ch), name=f"{name}_reshape")(x)
    return layers.Multiply(name=f"{name}_scale")([inputs, x])

def cbam_block(inputs, ratio=8, name="cbam"):
    ch = inputs.shape[-1]

    # ---- Channel Attention ----
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_c_gap")(inputs)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_c_gmp")(inputs)

    shared_fc1 = layers.Dense(ch // ratio, activation='relu', name=f"{name}_c_fc1")
    shared_fc2 = layers.Dense(ch, activation='sigmoid', name=f"{name}_c_fc2")

    avg_out = shared_fc2(shared_fc1(avg_pool))
    max_out = shared_fc2(shared_fc1(max_pool))
    ca = layers.Add(name=f"{name}_c_add")([avg_out, max_out])
    ca = layers.Reshape((1, 1, ch), name=f"{name}_c_reshape")(ca)
    x = layers.Multiply(name=f"{name}_c_scale")([inputs, ca])

    # ---- Spatial Attention ----
    avg_spatial = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_s_gap")(x)
    max_spatial = layers.GlobalMaxPooling2D(keepdims=True, name=f"{name}_s_gmp")(x)
    concat = layers.Concatenate(axis=-1, name=f"{name}_s_concat")([avg_spatial, max_spatial])
    sp = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', name=f"{name}_s_conv")(concat)
    return layers.Multiply(name=f"{name}_s_scale")([x, sp])

def attention_module(inputs):
    x = se_block(inputs, ratio=8, name="se")
    x = cbam_block(x, ratio=8, name="cbam")
    return x

# ===============================
# MODEL BUILD
# ===============================
def build_model(input_shape=(224, 224, 3), num_classes=4):
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Rescaling(1./255, name="rescale")(inp)

    base = applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        name="backbone_densenet121"
    )
    base.trainable = False

    x = base(x)
    x = attention_module(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inp, outputs=out, name="SE_CBAM_DenseNet")

# ===============================
# LOAD OR BUILD MODEL
# ===============================
if os.path.exists(LATEST_MODEL_PATH):
    print(f"🔁 Resuming from {LATEST_MODEL_PATH}")
    model = tf.keras.models.load_model(LATEST_MODEL_PATH, safe_mode=False)
else:
    print("🚀 Starting new training session...")
    model = build_model(IMG_SIZE + (3,), num_classes)

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    ModelCheckpoint(LATEST_MODEL_PATH, monitor='val_accuracy', save_best_only=False, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    CSVLogger(LOG_CSV, append=True)
]

# ===============================
# TRAINING STAGES
# ===============================
if not os.path.exists(LATEST_MODEL_PATH):
    print("\n🚀 Stage 1: Training with frozen DenseNet backbone")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
else:
    print("✅ Skipping Stage 1 — model already trained.")

# ===============================
# FINE-TUNE STAGE
# ===============================
print("\n🔧 Stage 2: Fine-tuning DenseNet top layers")
base = model.get_layer("backbone_densenet121")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST_LAYERS]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
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
