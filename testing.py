from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf

# Load the best saved model
best_model = tf.keras.models.load_model("attention_densenet_best.keras")

# Evaluate on the test set
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Get predictions
y_pred = np.argmax(best_model.predict(test_ds), axis=1)

# Get true labels (since it's a tf.data.Dataset)
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Classification report (precision, recall, F1-score)
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
