import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# CONFIG
# ===============================
TEST_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BEST_MODEL_PATH = "se_cbam_densenet_best_final.keras"
LOG_CSV = "training_log_improved.csv"

# ===============================
# LOAD TEST DATASET
# ===============================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("\n📁 Test Classes:", class_names)

# ===============================
# LOAD BEST MODEL
# ===============================
print("\n📦 Loading best model...")
model = tf.keras.models.load_model(BEST_MODEL_PATH, safe_mode=False)

# ===============================
# EVALUATE MODEL
# ===============================
print("\n🔍 Evaluating on TEST dataset...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)

print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

# ===============================
# TRUE LABELS & PREDICTIONS
# ===============================
y_true = np.concatenate([y.numpy() for _, y in test_ds.unbatch().batch(1000)], axis=0)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ===============================
# SAVE CLASSIFICATION REPORT
# ===============================
report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n📊 Classification Report:")
print(report_text)

with open("classification_report_test.txt", "w") as f:
    f.write(report_text)

print("📄 Saved: classification_report_test.txt")

# ===============================
# CONFUSION MATRIX HEATMAP
# ===============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.savefig("confusion_matrix_test.png", dpi=300)
plt.close()

print("🧩 Saved: confusion_matrix_test.png")

# ===============================
# SAVE PREDICTIONS CSV
# ===============================
df_preds = pd.DataFrame({
    "true_label": y_true,
    "predicted_label": y_pred
})
df_preds.to_csv("test_predictions.csv", index=False)

print("📑 Saved: test_predictions.csv")

# ===============================
# TRAINING CURVES (if log file exists)
# ===============================
if os.path.exists(LOG_CSV):
    df = pd.read_csv(LOG_CSV)

    # --- Accuracy Curve ---
    plt.figure(figsize=(8,6))
    plt.plot(df["accuracy"], label="Train Accuracy")
    plt.plot(df["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()

    # --- Loss Curve ---
    plt.figure(figsize=(8,6))
    plt.plot(df["loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    print("📊 Saved: accuracy_curve.png, loss_curve.png")
else:
    print("⚠️ No training log found — skipping curves.")

# ===============================
# GRAD-CAM GENERATION
# ===============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Save Grad-CAM images
GRADCAM_DIR = "gradcam_test_outputs"
os.makedirs(GRADCAM_DIR, exist_ok=True)

last_conv_layer = "conv5_block16_concat"  # DenseNet121 last conv

print("\n🔥 Generating Grad-CAM images...")
count = 10  # number of images

for idx, (img, label) in enumerate(test_ds.unbatch().take(count)):
    img_array = tf.expand_dims(img, axis=0)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    img_resized = cv2.resize(img.numpy().astype("uint8"), (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)

    save_path = os.path.join(GRADCAM_DIR, f"gradcam_{idx}.png")
    cv2.imwrite(save_path, overlay)

print(f"🔥 Saved Grad-CAM images in: {GRADCAM_DIR}")
print("\n✨ DONE — all test evaluation files generated!")
