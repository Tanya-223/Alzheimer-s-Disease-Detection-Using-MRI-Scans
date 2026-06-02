import tensorflow as tf

MODEL_PATH = "se_cbam_densenet_best_final.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

# Print summary
model.summary()
