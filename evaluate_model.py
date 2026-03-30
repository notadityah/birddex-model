#!/usr/bin/env python
"""
evaluate_model.py
-----------------
Evaluate a YOLO model on a dataset and optionally compare two models
side-by-side.  Designed to give contributors a quick, copy-pasteable
summary they can drop into a PR description.

Usage:
    # Evaluate a single model
    python evaluate_model.py runs/detect/train3/weights/best.pt --data dataset/data.yaml

    # Compare two models head-to-head
    python evaluate_model.py runs/detect/train5/weights/best.pt \
      --compare runs/detect/train3/weights/best.pt \
      --data dataset/data.yaml

    # Save results to JSON
    python evaluate_model.py best.pt --data dataset/data.yaml --save results.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def run_val(model_path: str, data_yaml: str) -> dict:
    """Run model.val() and return a structured results dict."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, plots=True, verbose=False)

    class_names = model.names  # {0: 'crow', 1: 'kingfisher', ...}
    per_class_map50 = metrics.box.maps  # per-class mAP50-95 array
    per_class = {}
    for idx, name in class_names.items():
        if idx < len(per_class_map50):
            per_class[name] = round(float(per_class_map50[idx]), 4)

    return {
        "model": str(Path(model_path).resolve()),
        "dataset": str(Path(data_yaml).resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        },
        "per_class_mAP50_95": per_class,
    }


def print_single(results: dict) -> None:
    """Print a clean summary table for a single model."""
    m = results["metrics"]
    print(f"\nModel: {results['model']}")
    print(f"Data:  {results['dataset']}")
    print()
    print(f"{'Metric':<20} {'Value':>10}")
    print("\u2500" * 32)
    for name, val in m.items():
        print(f"{name:<20} {val:>10.4f}")

    if results["per_class_mAP50_95"]:
        print(f"\n{'Per-class mAP50-95:'}")
        for cls, val in sorted(results["per_class_mAP50_95"].items()):
            print(f"  {cls:<25} {val:.4f}")
    print()


def print_comparison(results_a: dict, results_b: dict) -> None:
    """Print a side-by-side comparison table."""
    ma = results_a["metrics"]
    mb = results_b["metrics"]

    print(f"\nModel A: {results_a['model']}")
    print(f"Model B: {results_b['model']}  (baseline)")
    print()
    print(f"{'Metric':<20} {'A':>10} {'B':>10} {'Delta':>10}")
    print("\u2500" * 52)
    for key in ma:
        va, vb = ma[key], mb[key]
        delta = va - vb
        arrow = "\u2191" if delta > 0 else ("\u2193" if delta < 0 else " ")
        print(f"{key:<20} {va:>10.4f} {vb:>10.4f} {delta:>+10.4f} {arrow}")

    # Per-class comparison
    pca = results_a["per_class_mAP50_95"]
    pcb = results_b["per_class_mAP50_95"]
    all_classes = sorted(set(pca) | set(pcb))
    if all_classes:
        print(f"\n{'Per-class mAP50-95:'}")
        print(f"  {'Class':<25} {'A':>8} {'B':>8} {'Delta':>8}")
        print("  " + "\u2500" * 51)
        for cls in all_classes:
            va = pca.get(cls, float("nan"))
            vb = pcb.get(cls, float("nan"))
            delta = va - vb
            arrow = "\u2191" if delta > 0 else ("\u2193" if delta < 0 else " ")
            print(f"  {cls:<25} {va:>8.4f} {vb:>8.4f} {delta:>+8.4f} {arrow}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model and optionally compare against a baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        help="Path to the model to evaluate (e.g. runs/detect/train3/weights/best.pt)",
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--compare",
        metavar="BASELINE",
        help="Path to a baseline model for side-by-side comparison",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save results to a JSON file",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"Error: data.yaml not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating: {args.model}")
    results_a = run_val(args.model, args.data)

    if args.compare:
        if not Path(args.compare).exists():
            print(f"Error: baseline model not found: {args.compare}", file=sys.stderr)
            sys.exit(1)
        print(f"Evaluating baseline: {args.compare}")
        results_b = run_val(args.compare, args.data)
        print_comparison(results_a, results_b)
        output = {"model_a": results_a, "model_b": results_b}
    else:
        print_single(results_a)
        output = results_a

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output, indent=2))
        print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()
