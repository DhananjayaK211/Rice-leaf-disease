import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("artifacts/cnn_rice_leaf_transfer.keras")
class_names = ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_smut"]  # same as your training classes

# Path to your image
img_path = "data/test/healthy/groundnut.jpeg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # batch dimension

# Predict
pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]
Accuracy = np.max(pred)

print(f"Predicted Class: {pred_class} ({Accuracy *100:.2f}% Accuracy )")
