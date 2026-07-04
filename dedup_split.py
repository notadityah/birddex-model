#!/usr/bin/env python
"""
dedup_split.py
--------------
Turn a freshly-downloaded, possibly-duplicated image pool into a clean,
leakage-free train / val / test split for the bird detector.

Two data-quality problems it solves:

  1. Uniqueness  - iNaturalist returns near-duplicate photos (burst shots of the
     same bird, the same photo tagged under two species, etc.). These are removed
     via perceptual-hash (pHash) clustering, GLOBALLY across every species folder.

  2. Leakage     - the classic mistake is letting two photos of the same
     observation land on opposite sides of the train/test line. We split at the
     OBSERVATION level (from manifest.csv), so a whole observation stays in one
     split. Test is therefore genuinely held out for a fair old-vs-new comparison.

Reads:   dataset/manifest.csv   (written by download_bird_images.py)
Writes:  dataset/split.csv      (filename, species, split)  -> consumed by auto_label.py

Usage:
    python dedup_split.py                     # defaults: dataset/, 80/10/10
    python dedup_split.py --phash-thresh 5    # dedup aggressiveness (0 = identical only)
    python dedup_split.py --val-frac 0.1 --test-frac 0.1
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    import imagehash
    from PIL import Image
    _HAVE_PHASH = True
except ImportError:
    _HAVE_PHASH = False

SEED = 42
MANIFEST_NAME = "manifest.csv"
SPLIT_NAME = "split.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest rows; recompute any missing pHash from disk if possible."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _hash_from_row(row: dict, dataset_dir: Path) -> "imagehash.ImageHash | None":
    """Parse the stored pHash, or compute it from the file as a fallback."""
    raw = (row.get("phash") or "").strip()
    if raw:
        try:
            return imagehash.hex_to_hash(raw)
        except Exception:
            pass
    if not _HAVE_PHASH:
        return None
    img_path = dataset_dir / row["species"] / row["filename"]
    if not img_path.exists():
        return None
    try:
        with Image.open(img_path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception:
        return None


def _dedup(rows: list[dict], dataset_dir: Path, thresh: int) -> tuple[list[dict], int]:
    """
    Greedy global perceptual dedup. Iterates rows in a deterministic order and
    keeps an image only if its pHash is farther than `thresh` from every image
    already kept. Returns (kept_rows, dropped_count).
    """
    if not _HAVE_PHASH:
        print("[WARN] imagehash not installed - skipping perceptual dedup.")
        return rows, 0

    ordered = sorted(rows, key=lambda r: (r["species"], r["filename"]))
    kept: list[dict] = []
    kept_hashes: list = []
    dropped = 0

    for row in ordered:
        h = _hash_from_row(row, dataset_dir)
        if h is None:
            # Can't hash (missing file / unreadable) - keep it, let labeling decide.
            kept.append(row)
            continue
        is_dup = any((h - kh) <= thresh for kh in kept_hashes)
        if is_dup:
            dropped += 1
            continue
        kept.append(row)
        kept_hashes.append(h)

    return kept, dropped


def _observation_key(row: dict) -> str:
    """Group key for a row. Images with no obs_id are each their own group."""
    obs = (row.get("obs_id") or "").strip()
    return obs if obs else f"na::{row['species']}::{row['filename']}"


def _split_species(obs_ids: list[str], val_frac: float, test_frac: float,
                   rng: random.Random) -> dict[str, str]:
    """Assign whole observations of one species to train/val/test."""
    shuffled = obs_ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n - n_test >= 2 else 0

    assignment: dict[str, str] = {}
    for i, obs in enumerate(shuffled):
        if i < n_test:
            assignment[obs] = "test"
        elif i < n_test + n_val:
            assignment[obs] = "val"
        else:
            assignment[obs] = "train"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root (default: dataset)")
    parser.add_argument("--phash-thresh", type=int, default=5,
                        help="Max pHash Hamming distance to treat as duplicate (default: 5)")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Test fraction (default: 0.10)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        print("Run download_bird_images.py first (it writes manifest.csv).", file=sys.stderr)
        sys.exit(1)

    rows = _load_manifest(manifest_path)
    if not rows:
        print("Error: manifest is empty.", file=sys.stderr)
        sys.exit(1)

    # Keep only rows whose image still exists on disk.
    present = [r for r in rows if (dataset_dir / r["species"] / r["filename"]).exists()]
    missing = len(rows) - len(present)
    if missing:
        print(f"[WARN] {missing} manifest rows have no image on disk - ignored.")

    print(f"Loaded {len(present)} images across "
          f"{len(set(r['species'] for r in present))} species.")

    # ── 1. Global perceptual dedup ───────────────────────────────────────────
    kept, dropped = _dedup(present, dataset_dir, args.phash_thresh)
    print(f"Deduplication: dropped {dropped} near-duplicates, kept {len(kept)}.")

    # ── 2. Observation-level, per-species split ──────────────────────────────
    rng = random.Random(SEED)
    by_species: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        by_species[r["species"]].append(r)

    split_of_filename: dict[str, dict] = {}
    for species in sorted(by_species):
        srows = by_species[species]
        obs_to_rows: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            obs_to_rows[_observation_key(r)].append(r)

        assignment = _split_species(list(obs_to_rows), args.val_frac, args.test_frac, rng)
        for obs, obs_rows in obs_to_rows.items():
            split = assignment[obs]
            for r in obs_rows:
                split_of_filename[r["filename"]] = {
                    "filename": r["filename"],
                    "species": species,
                    "split": split,
                }

    # ── 3. Leakage assertion (observation ids disjoint across splits) ─────────
    obs_by_split: dict[str, set] = defaultdict(set)
    for r in kept:
        split = split_of_filename[r["filename"]]["split"]
        obs_by_split[split].add(_observation_key(r))
    test_obs = obs_by_split["test"]
    trainval_obs = obs_by_split["train"] | obs_by_split["val"]
    overlap = test_obs & trainval_obs
    assert not overlap, f"LEAKAGE: {len(overlap)} observations in both test and train/val"

    # ── 4. Write split.csv ───────────────────────────────────────────────────
    split_path = dataset_dir / SPLIT_NAME
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "species", "split"])
        writer.writeheader()
        for row in sorted(split_of_filename.values(), key=lambda r: (r["species"], r["filename"])):
            writer.writerow(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    counts = defaultdict(int)
    for r in split_of_filename.values():
        counts[r["split"]] += 1
    total = sum(counts.values())
    print(f"\n{'=' * 46}")
    print("  Split complete (observation-level, leakage-free)")
    print(f"  train : {counts['train']:>6}")
    print(f"  val   : {counts['val']:>6}")
    print(f"  test  : {counts['test']:>6}")
    print(f"  total : {total:>6}")
    print(f"  test observations disjoint from train/val: OK")
    print(f"  split.csv -> {split_path.resolve()}")
    print(f"{'=' * 46}")


if __name__ == "__main__":
    main()
