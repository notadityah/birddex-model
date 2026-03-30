#!/usr/bin/env python
# classify_bird.py
# Usage: python classify_bird.py <image_path> [--top 5]

import argparse
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

MODEL_ID = "chriamue/bird-species-classifier"


def main():
    # ── 1. Parse arguments ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Classify a bird image using the chriamue/bird-species-classifier HuggingFace model."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--top", type=int, default=5, help="Number of top predictions to show (default: 5)"
    )
    args = parser.parse_args()

    # ── 2. Load model & feature extractor ───────────────────────────────────────
    print(f"Loading model '{MODEL_ID}' ...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    # ── 3. Load & preprocess image ───────────────────────────────────────────────
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Could not load image '{args.image}': {e}")

    inputs = extractor(images=image, return_tensors="pt")

    # ── 4. Run inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on: {args.image}")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape: [1, num_classes]
    probs = torch.softmax(logits, dim=-1)[0]  # shape: [num_classes]

    # ── 5. Show top-N results ────────────────────────────────────────────────────
    top_probs, top_indices = torch.topk(probs, k=args.top)

    id2label = model.config.id2label  # label map built into the model config

    print(f"\n{'─'*45}")
    print(f"  Top {args.top} Predictions")
    print(f"{'─'*45}")
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), 1):
        label = id2label.get(idx, f"Class #{idx}")
        print(f"  {rank}. {label:<38} {prob * 100:6.2f}%")
    print(f"{'─'*45}")


if __name__ == "__main__":
    main()
