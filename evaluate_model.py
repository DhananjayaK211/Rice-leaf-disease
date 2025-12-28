import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def one_hot_labels(dataset, num_classes):
    def map_fn(x, y):
        y = tf.one_hot(y, num_classes)
        return x, y
    return dataset.map(map_fn)

def evaluate_model(model_path, data_dir=None, img_path=None, img_size=(224, 224), batch_size=32, save_csv=False):
    # Load trained CNN model
    model = tf.keras.models.load_model(model_path)
    class_names = ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_smut"]
    num_classes = len(class_names)

    if img_path:
        # Single image prediction
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = class_names[np.argmax(pred)]
        Accuracy = np.max(pred)
        print(f"Predicted Class: {pred_class} ({Accuracy *100:.2f}% Accuracy )")

    elif data_dir:
        # Folder evaluation
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )

        # Normalize and one-hot encode labels
        test_dataset = test_dataset.map(lambda x, y: (x/255.0, y))
        test_dataset = one_hot_labels(test_dataset, num_classes)

        # Evaluate
        loss, acc = model.evaluate(test_dataset)
        print(f"✅ Test Accuracy: {acc * 100:.2f}%")
        print(f"✅ Test Loss: {loss:.4f}")

        # Predictions
        true_classes = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
        true_classes = np.argmax(true_classes, axis=1)

        predictions = model.predict(test_dataset)
        predicted_classes = np.argmax(predictions, axis=1)

        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-Class Accuracy:")
        for cls, acc in zip(class_names, per_class_accuracy):
            print(f"{cls}: {acc*100:.2f}%")

        # Save predictions to CSV if needed
        if save_csv:
            image_files = []
            for folder, _, files in os.walk(data_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(folder, f))
            results = pd.DataFrame({
                "Image": image_files,
                "True": [class_names[c] for c in true_classes],
                "Predicted": [class_names[c] for c in predicted_classes]
            })
            results.to_csv("predictions.csv", index=False)
            print("\n✅ Predictions saved to predictions.csv")
    else:
        print("Error: Please provide either --data-dir or --image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained CNN model")
    parser.add_argument("--data-dir", type=str, help="Path to test dataset folder")
    parser.add_argument("--image", type=str, help="Path to a single image to predict")
    parser.add_argument("--save-csv", action="store_true", help="Save predictions to CSV (folder evaluation only)")
    args = parser.parse_args()

    evaluate_model(args.model, data_dir=args.data_dir, img_path=args.image, save_csv=args.save_csv)
