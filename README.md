# BirdDex

YOLO-based bird detection and species classification. Downloads images from iNaturalist, auto-labels them, trains a custom YOLO model, and evaluates results.

## Current Status

- **Model:** YOLOv11s fine-tuned for bird detection
- **Species:** 36 species (see [classes.txt](classes.txt))
- **Baseline:** `baseline/bird_detection.onnx` — the current production model to compare against

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd birddex-model
pip install -r requirements.txt

# 2. Download images for a species
python download_bird_images.py "Laughing Kookaburra" --count 80

# 3. Auto-label with bounding boxes
python auto_label.py

# 4. Review labels (visual QA)
python review_labels.py

# 5. Train
python train_model.py

# 6. Evaluate against baseline
python evaluate_model.py runs/detect/train/weights/best.pt \
  --compare baseline/bird_detection.onnx --data dataset/data.yaml
```

## Pipeline

> **Run scripts in this order.** Each step depends on the output of the previous one.

```
download_bird_images.py       Download images from iNaturalist
        |
   auto_label.py              Auto-generate YOLO bounding boxes
        |
  review_labels.py            Visual QA on labels
        |
  train_model.py              Train YOLOv11 on labeled dataset
        |
 evaluate_model.py            Measure mAP, precision, recall
        |
  yolo detect predict          Run inference on new images/videos
```

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
| `download_bird_images.py` | Download images from iNaturalist | `python download_bird_images.py "Galah" --count 80` |
| `auto_label.py` | Auto-generate YOLO bounding boxes | `python auto_label.py --conf-thresh 0.25` |
| `review_labels.py` | Visually review generated labels | `python review_labels.py --n 50` |
| `train_model.py` | Train YOLO model | `python train_model.py --epochs 100 --model yolo11m.pt` |
| `evaluate_model.py` | Evaluate and compare models | `python evaluate_model.py best.pt --compare baseline/bird_detection.onnx` |
| `classify_bird.py` | Classify species from cropped image | `python classify_bird.py bird.jpg --top 5` |
| `convert_model.py` | Export model to ONNX | `python convert_model.py best.pt` |
| `inspect_model.py` | Print model architecture and classes | `python inspect_model.py model.pt` |

All scripts support `--help` for full usage details.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding new species or improving model accuracy.
