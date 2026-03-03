import os
import shutil
import random
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = "BTAI_genImages_150"     # folder containing all your images
OUTPUT_DIR = "dataset"                # where to put train/val/test folders

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTS = {".png"}  # supported image types
# -----------------------------

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    source = Path(SOURCE_DIR)
    output = Path(OUTPUT_DIR)

    # Create destination folders
    for split in ["train", "val", "test"]:
        make_dir(output / "images" / split)
        make_dir(output / "labels" / split)

    # Get list of image files
    images = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    if len(images) == 0:
        print(f"No images found in {SOURCE_DIR}")
        return

    random.shuffle(images)
    n = len(images)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    splits = [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs)
    ]

    # Move/copy images and their label txt files if they exist
    for split_name, img_list in splits:
        print(f"Processing {split_name}: {len(img_list)} images")

        for img_path in img_list:
            # Copy image
            shutil.copy(img_path, output / "images" / split_name / img_path.name)

            # Copy label (YOLO expects same filename but .txt)
            label_path = img_path.with_suffix(".txt")
            dest_label = output / "labels" / split_name / label_path.name

            if label_path.exists():
                shutil.copy(label_path, dest_label)
            else:
                # If label doesn't exist, create an empty one (optional)
                open(dest_label, "w").close()

    print("Done! Dataset created at:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
