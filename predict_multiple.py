import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd

def predict_multiple(model_path, image_folder, img_size=(224, 224), save_csv=False):
    # Load model
    model = tf.keras.models.load_model(model_path)
    class_names = ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_smut"]

    # Collect all image paths
    img_files = []
    for root, _, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_files.append(os.path.join(root, f))

    results = []

    for img_path in img_files:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = class_names[np.argmax(pred)]
        Accuracy  = np.max(pred)

        print(f"{os.path.basename(img_path)} --> {pred_class} ({Accuracy *100:.2f}% confidence)")
        results.append({
            "Image": img_path,
            "Predicted": pred_class,
            "Accuracy ": Accuracy 
        })

    if save_csv:
        df = pd.DataFrame(results)
        df.to_csv("predictions_multiple.csv", index=False)
        print("\n✅ Predictions saved to predictions_multiple.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained CNN model")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images to predict")
    parser.add_argument("--save-csv", action="store_true", help="Save predictions to CSV")
    args = parser.parse_args()

    predict_multiple(args.model, args.image_folder, save_csv=args.save_csv)
