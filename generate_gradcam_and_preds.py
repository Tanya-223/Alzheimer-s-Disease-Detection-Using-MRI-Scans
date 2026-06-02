# generate_gradcam_and_preds.py
import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------- Configuration ----------
TEST_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Test"
BEST_MODEL_PATH = "se_cbam_densenet_best_final.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUT_DIR = "gradcam_outputs"
MAX_IMAGES = 200
# -----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Safe model loading
# ============================================================
def safe_load_model(path):
    try:
        print(f"Loading model (safe_mode=True): {path}")
        return load_model(path)
    except Exception as e:
        print("Normal load failed:", e)
        print("Retrying safe_mode=False ...")
        return load_model(path, safe_mode=False)


# ============================================================
# FIXED: Safe Conv layer finder
# ============================================================
def get_last_conv_layer(model):
    """
    Finds the last Conv2D layer in the model using layer.output.shape.
    Works with TF 2.13+ where .output_shape is deprecated.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            try:
                shape = tuple(layer.output.shape)
                if len(shape) == 4:  # (None, H, W, C)
                    print(f"🔍 Using Conv layer for Grad-CAM: {layer.name} | shape={shape}")
                    return layer.name
            except:
                continue
    raise ValueError("❌ No Conv2D layer found for Grad-CAM.")


# ============================================================
# Grad-CAM heatmap
# ============================================================
def make_gradcam_heatmap(img_tensor, model, last_conv_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.zeros((conv_outputs.shape[0], conv_outputs.shape[1]))

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_grads[i] * conv_outputs[:, :, i]

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


# ============================================================
# Overlay heatmap on image
# ============================================================
def overlay_heatmap_on_img(img, heatmap, alpha=0.4, colormap=plt.cm.jet):

    heatmap = np.uint8(255 * heatmap)
    color = colormap(heatmap)[:, :, :3]
    color = tf.image.resize(color, (img.shape[0], img.shape[1])).numpy()

    overlaid = (color * 255 * alpha + img * (1 - alpha))
    return np.uint8(np.clip(overlaid, 0, 255))


# ============================================================
# Normalize image
# ============================================================
def preprocess_image(x):
    return tf.cast(x, tf.float32) / 255.0


# ============================================================
# Main function
# ============================================================
def main():

    # 1) Load model
    model = safe_load_model(BEST_MODEL_PATH)
    model.summary()

    # 2) Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_ds.class_names
    print("Test classes:", class_names)

    # 3) Get last conv layer (fixed)
    try:
        last_conv = get_last_conv_layer(model)
    except ValueError as e:
        print("ERROR:", e)
        print("Available layers:", [layer.name for layer in model.layers])
        return

    # 4) Prepare CSV
    csv_path = os.path.join(OUT_DIR, "test_predictions.csv")
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["filename", "true_label", "pred_label", "pred_prob"])

    idx = 0

    # 5) Process test dataset
    for batch_imgs, batch_labels in test_ds:

        batch_imgs_proc = preprocess_image(batch_imgs)
        preds = model.predict(batch_imgs_proc, verbose=0)

        pred_labels = np.argmax(preds, axis=1)
        pred_probs = np.max(preds, axis=1)

        for i in range(batch_imgs.shape[0]):

            fname = f"img_{idx:06d}.png"
            true_label = int(batch_labels.numpy()[i])
            pred_label = int(pred_labels[i])
            pred_prob = float(pred_probs[i])

            writer.writerow([fname, class_names[true_label], class_names[pred_label], pred_prob])

            if idx < MAX_IMAGES:
                img = batch_imgs.numpy()[i]
                img_input = tf.expand_dims(preprocess_image(img), 0)

                heatmap = make_gradcam_heatmap(img_input, model, last_conv, pred_label)
                overlaid = overlay_heatmap_on_img(img, heatmap)

                out_path = os.path.join(
                    OUT_DIR,
                    f"gradcam_{idx:06d}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png"
                )
                plt.imsave(out_path, overlaid)

            idx += 1

    f.close()

    print("✅ Saved:", csv_path)
    print("🔥 Grad-CAM images saved to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
