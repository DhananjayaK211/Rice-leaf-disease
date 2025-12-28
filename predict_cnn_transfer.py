import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Command-line argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Predict rice leaf disease using CNN")
parser.add_argument('--image', required=True, help='Path to input image')
args = parser.parse_args()
img_path = args.image

# -----------------------------
# Load trained model
# -----------------------------
model_path = 'artifacts/cnn_mobilenet_transfer.h5'  # fine-tuned model
model = load_model(model_path)

# -----------------------------
# Class labels
# -----------------------------
# Make sure this matches the classes you trained on
class_labels = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut', 'Healthy']

# -----------------------------
# Load and preprocess image
# -----------------------------
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x / 255.0  # rescale
x = np.expand_dims(x, axis=0)

# -----------------------------
# Make prediction
# -----------------------------
preds = model.predict(x)
pred_class = class_labels[np.argmax(preds)]
Accuracy  = np.max(preds) * 100

print(f"Predicted Class: {pred_class}")
print(f"Accuracy : {Accuracy :.2f}%")
