import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- same parameters used before ---
img_size = 128
batch_size = 32
base_path =r"C:\Users\Tanya\Desktop\OASIS_dataset\Data" # 🔹 put your actual dataset path here

# --- Recreate validation generator (must match original preprocessing) ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    base_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- Load saved model ---
model = tf.keras.models.load_model('best_model.keras')

# --- Evaluate accuracy ---
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"📉 Final Validation Loss: {val_loss:.4f}")
