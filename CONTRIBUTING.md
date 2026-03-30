# Contributing to BirdDex

There are two main ways to contribute: adding new bird species or improving detection accuracy on existing species.

## Path 1: Add a New Species

1. Download training images (this creates the species folder that `auto_label.py` discovers):
   ```bash
   python download_bird_images.py "Species Name" --count 80
   ```
2. Add the species name (lowercase, underscores) to `classes.txt` for reference tracking
3. Auto-label the full dataset (all species):
   ```bash
   python auto_label.py
   ```
4. Review labels to verify quality:
   ```bash
   python review_labels.py
   ```
5. Retrain from scratch with all species (prevents catastrophic forgetting):
   ```bash
   python train_model.py
   ```
6. Evaluate and compare against the previous model:
   ```bash
   python evaluate_model.py runs/detect/train/weights/best.pt \
     --compare <previous_model.pt> --data dataset/data.yaml
   ```
7. Include the comparison output in your PR description

## Path 2: Improve Accuracy (Same Species)

- Tune hyperparameters (epochs, model size, image size, batch size)
- Add more training data for underperforming species
- Try different base models (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`)

Same evaluation step at the end: run `evaluate_model.py --compare` and include the output in your PR.

## PR Requirements

- **Must** include `evaluate_model.py` comparison output in the PR description
- New model's mAP50 must be >= baseline mAP50
- If adding species: update `classes.txt`
- Do not commit model weight files (`.pt`, `.onnx`) — they are gitignored

## Tips

- Use `python review_labels.py` to catch bad auto-labels before training
- Lower `--conf-thresh` in `auto_label.py` if too many images are being skipped
- Start with `yolo11s.pt` (balanced speed/accuracy), move to `yolo11m.pt` if needed
- Check `unlabeled.txt` after auto-labeling for images that need manual review
