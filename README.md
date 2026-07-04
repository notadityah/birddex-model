# BirdDex

YOLO-based bird detection and species classification. Downloads images from iNaturalist, auto-labels them, trains a custom YOLO model, and evaluates results.

## Current Status

- **Model:** YOLOv11s fine-tuned for bird detection
- **Species:** 36 species (see [classes.txt](classes.txt))
- **Baseline:** `baseline/bird_detection.onnx` — the current production model to compare against

## Results

Both models evaluated on the same frozen, held-out test set (`dataset/test.yaml`) built by
`dedup_split.py` — deduplicated and split at the observation level, so no test image (or
another photo of the same bird sighting) was ever seen during training. Full per-class
numbers are in [results.json](results.json).

| Metric | Baseline (`bird_detection.onnx`) | Retrained (`best.pt`) | Δ |
|--------|------:|------:|------:|
| mAP50 | 0.377 | 0.744 | +0.367 |
| mAP50-95 | 0.299 | 0.649 | +0.350 |
| Precision | 0.577 | 0.824 | +0.247 |
| Recall | 0.321 | 0.644 | +0.322 |

Both models are the same YOLOv11s architecture — the gains come entirely from better data:
more images per species (250 vs 80), perceptual-hash deduplication, a leakage-free split,
and multi-bird auto-labeling (every bird in a photo gets a box, not just the top-1).

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd birddex-model
pip install -r requirements.txt

# 2. Download images (writes dataset/manifest.csv)
python download_bird_images.py "Laughing Kookaburra" --count 250   # one species
python download_bird_images.py --file birds_list.txt --count 250   # all 36 species

# 3. Deduplicate + build a leakage-free train/val/test split
python dedup_split.py

# 4. Auto-label with bounding boxes (writes data.yaml + test.yaml)
python auto_label.py

# 5. Review labels (visual QA)
python review_labels.py

# 6. Train (on Colab GPU, or locally with --device mps)
python train_model.py

# 7. Evaluate BOTH models on the frozen held-out test set
python evaluate_model.py runs/detect/train/weights/best.pt \
  --compare baseline/bird_detection.onnx --data dataset/test.yaml
```

## Pipeline

> **Run scripts in this order.** Each step depends on the output of the previous one.

```
download_bird_images.py       Download images from iNaturalist (+ manifest.csv)
        |
   dedup_split.py             Perceptual dedup + observation-level train/val/test split
        |
   auto_label.py              Auto-generate YOLO boxes (all birds), data.yaml + test.yaml
        |
  review_labels.py            Visual QA on labels
        |
  train_model.py              Train YOLOv11 on labeled dataset
        |
 evaluate_model.py            Measure mAP on the held-out test set (data: test.yaml)
        |
  yolo detect predict          Run inference on new images/videos
```

**Fair comparison:** `dedup_split.py` holds out whole iNaturalist *observations* as a
test set that never touches training, so `evaluate_model.py --compare ... --data
dataset/test.yaml` scores the old and new models on identical, leakage-free data.

## Running Inference

Use the built-in ultralytics CLI:

```bash
# Single image
yolo detect predict model=best.pt source=path/to/image.jpg

# Video
yolo detect predict model=best.pt source=path/to/video.mp4

# Folder of images
yolo detect predict model=best.pt source=path/to/folder/
```

## Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `download_bird_images.py` | Download images from iNaturalist (+ manifest.csv) | `python download_bird_images.py "Galah" --count 250` |
| `dedup_split.py` | Perceptual dedup + leakage-free train/val/test split | `python dedup_split.py --phash-thresh 5` |
| `auto_label.py` | Auto-generate YOLO bounding boxes (all birds) | `python auto_label.py --conf-thresh 0.25` |
| `review_labels.py` | Visually review generated labels | `python review_labels.py --n 50` |
| `train_model.py` | Train YOLO model | `python train_model.py --epochs 100 --model yolo11m.pt` |
| `evaluate_model.py` | Evaluate and compare models | `python evaluate_model.py best.pt --compare baseline/bird_detection.onnx` |
| `classify_bird.py` | Classify species from cropped image | `python classify_bird.py bird.jpg --top 5` |
| `convert_model.py` | Export model to ONNX | `python convert_model.py best.pt` |
| `inspect_model.py` | Print model architecture and classes | `python inspect_model.py model.pt` |

All scripts support `--help` for full usage details.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding new species or improving model accuracy.
