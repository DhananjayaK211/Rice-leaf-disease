import os
import shutil
import random

# Paths
data_dir = "data/dataset"
output_dir = "data"

# Only keep folders (ignore files like CSV, JPG)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        split_dir = os.path.join(output_dir, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)
            if not os.path.exists(dst):  # avoid duplicate copies
                shutil.copy(src, dst)

print("✅ Dataset split into train/val/test successfully!")
