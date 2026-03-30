import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX format.")
    parser.add_argument("model", nargs="?", default="best.pt", help="Path to the .pt model (default: best.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    if args.imgsz % 32 != 0:
        print(f"Error: --imgsz must be a multiple of 32, got {args.imgsz}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.model)
    try:
        model.export(
            format="onnx",
            simplify=True,
            opset=17,
            dynamic=args.dynamic,
            imgsz=args.imgsz,
        )
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Export complete! Model saved as {args.model.replace('.pt', '.onnx')}")


if __name__ == "__main__":
    main()
