# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

This is a **bird detection and classification system** that uses YOLOv8/v11 for object detection and HuggingFace models for species classification. The pipeline includes dataset preparation, model training, evaluation, and inference.

**Current trained models:**

- `model.pt` - PyTorch YOLOv8 model for bird detection
- `model.onnx` - ONNX-formatted version of the model for inference
- Config supports 400+ bird species (in `config.json`)
- 36 species in `classes.txt` (current training target)

## Quick Commands

### Setup & Dependencies

```bash
pip install -r requirements.txt
```

### Core Workflows

**1. Download Images from iNaturalist** (writes `dataset/manifest.csv`)

```bash
python download_bird_images.py "Laughing Kookaburra" --count 250
python download_bird_images.py --file birds_list.txt --count 250
python download_bird_images.py "Galah" --count 250 --output my_dataset
```

**2. Deduplicate + Split** (perceptual dedup, observation-level train/val/test)

```bash
python dedup_split.py                    # defaults: 80/10/10, phash-thresh 5
python dedup_split.py --phash-thresh 8   # more aggressive dedup
# Output: dataset/split.csv (filename -> split), leakage-free by observation id
```

**3. Prepare Dataset (Auto-labeling)**

```bash
python auto_label.py                              # defaults (yolo11x.pt, reads split.csv)
python auto_label.py --conf-thresh 0.30           # override confidence
# Output: dataset/images/{train,val,test}/ + labels/, plus data.yaml AND test.yaml
# Keeps EVERY bird box above --conf-thresh (multi-bird images fully labeled)
```

**4. Review Auto-Generated Labels**

```bash
python review_labels.py           # Review 20 random images
python review_labels.py --n 50    # Review 50 images
python review_labels.py --split val  # Review validation set only
```

**5. Train Model**

```bash
python train_model.py                                    # defaults (yolo11s.pt, 50 epochs)
python train_model.py --epochs 100 --model yolo11m.pt    # overrides
python train_model.py --help                             # all flags
```

**6. Evaluate Model**

```bash
# Single model evaluation on the frozen held-out test set
python evaluate_model.py runs/detect/train/weights/best.pt --data dataset/test.yaml

# Compare new vs current model head-to-head (same test set = fair)
python evaluate_model.py new_model.pt --compare baseline/bird_detection.onnx --data dataset/test.yaml

# Save results to JSON
python evaluate_model.py best.pt --data dataset/test.yaml --save results.json
```

> Always evaluate with `--data dataset/test.yaml` (not `data.yaml`) so the score is on
> observations never seen during training. `data.yaml`'s val set is used only for training.

**7. Run Inference (using ultralytics CLI)**

```bash
yolo detect predict model=best.pt source=path/to/image.jpg
yolo detect predict model=best.pt source=path/to/video.mp4
yolo detect predict model=best.pt source=path/to/folder/
```

**8. Classify Bird Species**

```bash
python classify_bird.py <image_path> [--top 5]
# Uses chriamue/bird-species-classifier from HuggingFace
```

**9. Export & Inspect Models**

```bash
python convert_model.py best.pt              # Export to ONNX
python convert_model.py best.pt --dynamic    # With dynamic batch size
python inspect_model.py model.pt             # Inspect .pt model
python inspect_model.py model.onnx           # Inspect .onnx model
```

## Architecture & Data Flow

### Main Components

1. **Data Preparation Pipeline**
   - `download_bird_images.py` - Fetches images from iNaturalist API (public, no key needed); records observation id + perceptual hash to `dataset/manifest.csv`
   - `dedup_split.py` - Drops near-duplicate photos (pHash) and splits at the observation level into train/val/test (80/10/10), guaranteeing the test set shares no observation with train/val (no leakage). Emits `dataset/split.csv`
   - `auto_label.py` - Auto-generates bounding boxes using a YOLO detector (default `yolo11x.pt`, COCO bird class 14); keeps every box above `--conf-thresh` (default 0.25); honors `split.csv`

2. **Training & Model Management**
   - `train_model.py` - Trains YOLOv11s on custom dataset (all params configurable via CLI)
   - Uses YOLO format: `dataset/{images,labels}/{train,val,test}/`
   - Configuration: `data.yaml` (paths, class names, count)
   - Outputs best model to `runs/detect/train/weights/best.pt`

3. **Evaluation & Comparison**
   - `evaluate_model.py` - Runs `model.val()` and prints mAP50, mAP50-95, precision, recall
   - `--compare` flag for side-by-side model comparison with delta table
   - Outputs can be saved to JSON for record-keeping

4. **Inference**
   - Use `yolo detect predict` (ultralytics CLI) for image/video inference
   - Also: `yolo detect train/val/export` for other ultralytics operations

5. **Species Classification**
   - `classify_bird.py` - Classifies detected birds using HuggingFace EfficientNet-b2
   - Supports 400+ bird species from `config.json`

6. **Quality Assurance**
   - `review_labels.py` - Visual QA tool for bounding boxes
   - Randomly samples labeled images, draws boxes with colors per class

### Dataset Format

```
dataset/
├── <species_name>/          # Raw image folders
│   ├── bird1.jpg
│   └── ...
├── manifest.csv             # species, obs_id, filename, phash (download_bird_images.py)
├── split.csv                # filename -> train/val/test  (dedup_split.py)
├── images/
│   ├── train/              # ~80% of observations
│   ├── val/                # ~10% of observations
│   └── test/               # ~10% held out, leakage-free
├── labels/
│   ├── train/              # YOLO .txt format (one line per bird box)
│   ├── val/
│   └── test/
├── data.yaml               # train + val (for training)
└── test.yaml               # test set (for evaluate_model.py)
```

**YOLO Label Format** (`labels/train/image.txt`):

```
<class_id> <cx> <cy> <w> <h>
```

Where cx, cy, w, h are normalized (0-1) relative to image dimensions.

### Training Tips (from HOW_TO_TRAIN.md)

- **Adding new bird species**: Retraining from scratch (with yolo11s.pt) is recommended to avoid "catastrophic forgetting"
- **Fine-tuning existing model**: Only viable if class count stays exactly the same
- **Epochs**: Default 50 (adjust via `--epochs`)
- **Image size**: Default 640 (adjust via `--imgsz`, must be multiple of 32)
- **Model sizes**: `yolo11n.pt` (fastest) / `yolo11s.pt` (default) / `yolo11m.pt` (most accurate)

## Key Dependencies

- `ultralytics` - YOLOv8/v11 framework (includes CLI: `yolo detect train/val/predict/export`)
- `supervision` - Annotation & tracking utilities
- `opencv-python` - Image/video I/O and processing
- `transformers` - HuggingFace model loading
- `torch` / `torchvision` - Deep learning backend
- `numpy`, `matplotlib` - Numerical/visualization
- `requests` - HTTP requests (iNaturalist API)
- `Pillow` - Image processing
- `imagehash` - Perceptual hashing for dataset deduplication (dedup_split.py)
- `onnx` - ONNX model inspection

## Important Implementation Details

### Model Configuration

- `config.json` - EfficientNet-b2 config with 400+ bird species mapping (id2label)
- `classes.txt` - List of species the YOLO model is trained to detect (currently 36)

### Auto-labeling (auto_label.py)

- Uses a YOLO detector to find birds (COCO class 14); default `yolo11x.pt` for quality
- Keeps EVERY box above `--conf-thresh` (default 0.25) - multi-bird photos fully labeled
- Reads `dataset/split.csv` for train/val/test placement (falls back to random val split)
- Skips images where no bird detected (logged to `unlabeled.txt`)
- All config via CLI flags: `--conf-thresh`, `--val-split`, `--detector-model`, `--dataset-dir`
- Writes both `data.yaml` (train+val) and `test.yaml` (held-out test)

### Evaluation (evaluate_model.py)

- Access metrics via `model.val()`: `metrics.box.map50`, `metrics.box.map`, `metrics.box.mp`, `metrics.box.mr`
- Per-class metrics from `metrics.box.maps` indexed by `model.names`
- Comparison mode prints delta table suitable for PR descriptions

## Troubleshooting

- **Out of GPU memory**: Reduce batch size (`--batch`) or use smaller model (`--model yolo11n.pt`)
- **Auto-label skipping many images**: Lower confidence with `--conf-thresh 0.15`
- **Model not detecting birds**: Check confidence threshold; verify model file exists
