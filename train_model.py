import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def train(args):
    # OPTION 1 (default): Start from a pretrained base model and train on your
    # FULL dataset (old + new birds).  This prevents "catastrophic forgetting".
    #
    # OPTION 2: To continue training an existing model (only if classes remain
    # exactly the same), pass --model model.pt instead.
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Starting training...")
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
        )
        print("Training complete!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure your 'data.yaml' path is correct and your dataset is properly formatted.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on a bird detection dataset.",
    )
    parser.add_argument("--model", default="yolo11s.pt", help="Base model to start from (default: yolo11s.pt)")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml (default: dataset/data.yaml)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size, must be multiple of 32 (default: 640)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size, -1 for auto (default: -1)")
    parser.add_argument("--device", default="0", help="Device to train on: 0, 1, cpu, etc. (default: 0)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: data.yaml not found: {args.data}", file=sys.stderr)
        print("Run auto_label.py first to generate the dataset and data.yaml.")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
