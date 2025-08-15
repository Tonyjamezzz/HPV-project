import argparse
import csv
import random
from pathlib import Path
from PIL import Image
import numpy as np

def process_image(src: Path, dst: Path, size: int) -> None:
    """Resize image and normalize pixel values."""
    img = Image.open(src).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype("float32") / 255.0
    img = Image.fromarray((arr * 255).astype("uint8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)

def split_list(items, val_ratio):
    random.shuffle(items)
    n_val = int(len(items) * val_ratio)
    return items[n_val:], items[:n_val]

def main():
    parser = argparse.ArgumentParser(description="Preprocess cervical cell images")
    parser.add_argument("--raw-dir", default="data/raw/Herlev", help="Directory with raw images")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    random.seed(42)
    raw_path = Path(args.raw_dir)
    out_path = Path(args.processed_dir)
    for split in ["train", "val", "test"]:
        (out_path / split).mkdir(parents=True, exist_ok=True)

    records = {"train": [], "val": [], "test": []}

    # Process test images
    for class_dir in (raw_path / "test").iterdir():
        label = class_dir.name
        for img_path in class_dir.glob("*"):
            dst = out_path / "test" / label / img_path.name
            process_image(img_path, dst, args.size)
            records["test"].append((str(dst.relative_to(out_path)), label))

    # Process train images and split into train/val
    for class_dir in (raw_path / "train").iterdir():
        label = class_dir.name
        images = list(class_dir.glob("*"))
        train_imgs, val_imgs = split_list(images, args.val_ratio)
        for img_path in train_imgs:
            dst = out_path / "train" / label / img_path.name
            process_image(img_path, dst, args.size)
            records["train"].append((str(dst.relative_to(out_path)), label))
        for img_path in val_imgs:
            dst = out_path / "val" / label / img_path.name
            process_image(img_path, dst, args.size)
            records["val"].append((str(dst.relative_to(out_path)), label))

    # Write label files
    for split, rows in records.items():
        with open(out_path / f"{split}_labels.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label"])
            writer.writerows(rows)

if __name__ == "__main__":
    main()
