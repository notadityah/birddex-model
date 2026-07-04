#!/usr/bin/env python
# auto_label.py
# Automatically labels bird images with bounding boxes using a pretrained YOLO
# bird detector (COCO class 14 = bird). Every bird box above the confidence
# threshold is kept (not just the top-1), so multi-bird photos are fully labeled.
# Images where no bird is detected are skipped and logged to 'unlabeled.txt'.
#
# The train/val/test assignment is read from dataset/split.csv (produced by
# dedup_split.py) so the split is deduplicated and leakage-free. If split.csv is
# absent, it falls back to a random per-species train/val split.
#
# Output structure (YOLO format, ready for train_model.py):
#   dataset/images/{train,val,test}/   dataset/labels/{train,val,test}/
#   dataset/data.yaml   (train + val, for training)
#   dataset/test.yaml   (held-out test, for evaluate_model.py)

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

YOLO_BIRD_CLS = 14  # COCO class id for "bird"
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAME = "split.csv"


def load_split(dataset_dir: Path) -> dict[str, str]:
    """Return {filename: split} from split.csv, or {} if it doesn't exist."""
    split_path = dataset_dir / SPLIT_NAME
    if not split_path.exists():
        return {}
    with open(split_path, newline="", encoding="utf-8") as f:
        return {row["filename"]: row["split"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label bird images using a pretrained YOLO detector.",
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Root dataset directory (default: dataset)")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Min detection confidence (default: 0.25)")
    parser.add_argument("--val-split", type=float, default=0.10, help="Val fraction when split.csv is absent (default: 0.10)")
    parser.add_argument("--detector-model", default="yolo11x.pt", help="Pretrained detector model (default: yolo11x.pt)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = dataset_dir
    val_split = args.val_split
    conf_thresh = args.conf_thresh
    detector_model = args.detector_model

    random.seed(SEED)

    # ── Discover species folders ─────────────────────────────────────────────
    species_dirs = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("images", "labels")
    ])

    if not species_dirs:
        raise RuntimeError(f"No species folders found in '{dataset_dir}'")

    # Map folder name -> class index (alphabetical order)
    classes = [d.name for d in species_dirs]
    cls_to_idx = {name: idx for idx, name in enumerate(classes)}
    num_classes = len(classes)

    print(f"Found {num_classes} species:")
    for i, c in enumerate(classes):
        print(f"  {i:>2}: {c}")

    # ── Load the split produced by dedup_split.py ────────────────────────────
    split_map = load_split(dataset_dir)
    have_test = False
    if split_map:
        print(f"\nUsing split.csv ({len(split_map)} images) for train/val/test assignment.")
        have_test = "test" in split_map.values()
    else:
        print("\n[WARN] split.csv not found - falling back to random train/val split "
              "(no test set, no dedup). Run dedup_split.py for a proper split.")

    # ── Load YOLO detector ───────────────────────────────────────────────────
    print(f"\nLoading detector '{detector_model}' ...")
    detector = YOLO(detector_model)

    # ── Prepare output directories ───────────────────────────────────────────
    splits = ("train", "val", "test") if have_test else ("train", "val")
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Label all images ─────────────────────────────────────────────────────
    labeled = 0
    boxes_written = 0
    skipped = 0
    skipped_list = []

    for species_dir in species_dirs:
        cls_idx = cls_to_idx[species_dir.name]
        images = [f for f in species_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]

        if not images:
            continue

        # Determine per-image split.
        if split_map:
            images = [f for f in images if f.name in split_map]
        else:
            random.shuffle(images)
            n_val = max(1, int(len(images) * val_split))
            fallback_val = set(f.name for f in images[:n_val])

        print(f"\n[{species_dir.name}] {len(images)} images")

        for img_path in images:
            if split_map:
                split = split_map[img_path.name]
            else:
                split = "val" if img_path.name in fallback_val else "train"

            # ── Run detection ────────────────────────────────────────────────
            results = detector.predict(
                source=str(img_path),
                conf=conf_thresh,
                classes=[YOLO_BIRD_CLS],
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                skipped += 1
                skipped_list.append(str(img_path))
                continue

            # ── Copy image ───────────────────────────────────────────────────
            dst_img = output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # ── Write YOLO label: one line per detected bird box ──────────────
            dst_lbl = output_dir / "labels" / split / (img_path.stem + ".txt")
            with open(dst_lbl, "w") as f:
                for box in boxes.xywhn.tolist():  # [cx, cy, w, h] normalised
                    f.write(f"{cls_idx} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                    boxes_written += 1

            labeled += 1

    # ── Write unlabeled log ──────────────────────────────────────────────────
    unlabeled_file = output_dir / "unlabeled.txt"
    if skipped_list:
        unlabeled_file.write_text("\n".join(skipped_list))
        print(f"\nImages with no bird detected saved to '{unlabeled_file}' for manual review.")

    # ── Write data.yaml (train + val) and test.yaml (held-out test) ──────────
    abs_dataset = output_dir.resolve().as_posix()

    def write_yaml(path: Path, val_split_name: str, header: str):
        content = f"""# {header}
path: {abs_dataset}
train: images/train
val:   images/{val_split_name}

nc: {num_classes}
names:
"""
        for cls in classes:
            content += f"  - {cls}\n"
        path.write_text(content)

    data_yaml = output_dir / "data.yaml"
    write_yaml(data_yaml, "val", "Auto-generated by auto_label.py (training)")

    test_yaml = None
    if have_test:
        test_yaml = output_dir / "test.yaml"
        write_yaml(test_yaml, "test", "Auto-generated by auto_label.py (frozen test set)")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = labeled + skipped
    print(f"\n{'=' * 50}")
    print(f"  Labeling complete!")
    print(f"  Labeled  : {labeled:>5} / {total} images ({boxes_written} boxes)")
    print(f"  Skipped  : {skipped:>5} / {total} images (no bird detected)")
    print(f"  data.yaml: {data_yaml.resolve()}")
    if test_yaml:
        print(f"  test.yaml: {test_yaml.resolve()}")
    print(f"{'=' * 50}")
    if skipped:
        print(f"\n  Review '{unlabeled_file}' and label manually or remove those images.")


if __name__ == "__main__":
    main()
