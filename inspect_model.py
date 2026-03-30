import argparse
import sys
from pathlib import Path


def inspect_pt(model_path: str):
    from ultralytics import YOLO
    model = YOLO(model_path)
    print("-" * 30)
    print("Model Architecture Summary:")
    print("-" * 30)
    model.info()
    print("\n" + "-" * 30)
    print("Classes (What it detects):")
    print("-" * 30)
    print(model.names)


def inspect_onnx(model_path: str):
    import onnx
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    for inp in model.graph.input:
        print("Input:", inp.name, inp.type.tensor_type.shape)
    for out in model.graph.output:
        print("Output:", out.name, out.type.tensor_type.shape)


def main():
    parser = argparse.ArgumentParser(description="Inspect a YOLO model (.pt or .onnx).")
    parser.add_argument("model", nargs="?", default="model.pt", help="Path to model file (default: model.pt)")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"Error: model not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".onnx":
        inspect_onnx(args.model)
    else:
        inspect_pt(args.model)


if __name__ == "__main__":
    main()
