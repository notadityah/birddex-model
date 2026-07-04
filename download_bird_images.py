"""
download_bird_images.py
-----------------------
Download bird images by species name using the iNaturalist public API.
Images come from research-grade, community-verified observations — ideal
for training bird detection / classification models.

Usage examples:
    # Download 80 images of a single species
    python download_bird_images.py "Laughing Kookaburra" --count 80

    # Download multiple species at once
    python download_bird_images.py "Rainbow Lorikeet" "Australian Magpie" --count 100

    # Use a text file with one bird name per line
    python download_bird_images.py --file birds.txt --count 80

    # Change the output directory
    python download_bird_images.py "Galah" --count 60 --output my_dataset

No API key required — uses the free iNaturalist Observations API.
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

try:
    import imagehash
    from PIL import Image
    _HAVE_PHASH = True
except ImportError:  # pragma: no cover - optional dependency guard
    _HAVE_PHASH = False


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

INAT_API = "https://api.inaturalist.org/v1/observations"

HEADERS = {
    "User-Agent": "bird-detection-downloader/1.0 (contact: local-script)",
    "Accept": "application/json",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# iNaturalist returns at most 200 per page
PAGE_SIZE = 200

# Manifest columns (one row per downloaded image). Consumed by dedup_split.py.
MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = ["species", "obs_id", "filename", "phash"]


# ─────────────────────────────────────────────
#  iNaturalist photo URL fetcher
# ─────────────────────────────────────────────

def _inaturalist_photo_urls(taxon_name: str, max_results: int) -> list[tuple[int, str]]:
    """
    Fetch photos from iNaturalist research-grade observations.
    Returns a list of (observation_id, medium_url) pairs (~800px wide).

    The observation id is carried through so downstream splitting can keep every
    photo of the same observation on one side of the train/val/test boundary.
    """
    pairs: list[tuple[int, str]] = []
    seen_urls: set[str] = set()
    page = 1

    while len(pairs) < max_results:
        need = max_results - len(pairs)
        per_page = min(PAGE_SIZE, need + 20)  # small buffer

        params = {
            "taxon_name": taxon_name,
            "quality_grade": "research",   # community-verified IDs only
            "photos": "true",
            "per_page": per_page,
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }

        data = None
        last_exc = None
        for attempt in range(3):  # retry transient API/network errors
            try:
                resp = requests.get(INAT_API, params=params, headers=HEADERS, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"iNaturalist API error after retries: {last_exc}") from last_exc

        results = data.get("results", [])
        if not results:
            break  # no more pages

        for obs in results:
            obs_id = obs.get("id")
            for photo in obs.get("photos", []):
                raw_url = photo.get("url", "")
                if not raw_url:
                    continue
                # Swap square thumbnail for medium (better quality for training)
                med_url = re.sub(r"/square\.", "/medium.", raw_url)
                med_url = re.sub(r"/square\b", "/medium", med_url)
                if med_url in seen_urls:
                    continue
                seen_urls.add(med_url)
                pairs.append((obs_id, med_url))
                if len(pairs) >= max_results:
                    break
            if len(pairs) >= max_results:
                break

        total = data.get("total_results", 0)
        if page * per_page >= total:
            break  # exhausted all pages

        page += 1
        time.sleep(0.5)  # be polite to iNaturalist

    return pairs[:max_results]


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _safe_folder_name(bird_name: str) -> str:
    """Convert a bird name into a safe directory name."""
    return re.sub(r"[^a-z0-9_]", "_", bird_name.strip().lower())


def _url_extension(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext.lower()


def _compute_phash(image_path: Path) -> str:
    """Perceptual hash of an image as a hex string, or '' if unavailable."""
    if not _HAVE_PHASH:
        return ""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img.convert("RGB")))
    except Exception:
        return ""


def _append_manifest(manifest_path: Path, rows: list[dict]) -> None:
    """Append rows to the dataset manifest, writing a header if new."""
    if not rows:
        return
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────
#  Main downloader
# ─────────────────────────────────────────────

def download_images(
    bird_name: str,
    count: int,
    output_dir: Path,
    skip_existing: bool = True,
    folder_name: str | None = None,
) -> tuple[int, int]:
    """
    Download `count` images for `bird_name` into output_dir/<folder>/.
    If `folder_name` is given it is used verbatim (after sanitizing); otherwise
    it is derived from `bird_name`. Returns (downloaded, skipped) counts.
    """
    folder_name = _safe_folder_name(folder_name or bird_name)
    save_dir = output_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME

    print(f"\nFetching: '{bird_name}' -> {save_dir}")

    try:
        pairs = _inaturalist_photo_urls(bird_name, max_results=count + 30)
    except Exception as exc:
        print(f"   [ERROR] Failed to fetch URLs: {exc}")
        return 0, 0

    if not pairs:
        print("   [WARN] No photos found on iNaturalist for this species.")
        return 0, 0

    downloaded = 0
    skipped = 0
    manifest_rows: list[dict] = []

    for obs_id, url in pairs:
        if downloaded >= count:
            break

        ext = _url_extension(url)
        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".jpg"

        # Deterministic filename based on URL hash (avoids duplicates on re-run)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        filename = save_dir / f"{folder_name}_{url_hash}{ext}"

        if skip_existing and filename.exists():
            skipped += 1
            continue

        try:
            img_resp = requests.get(url, headers=HEADERS, timeout=20, stream=True)
            img_resp.raise_for_status()

            content_type = img_resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                continue

            with open(filename, "wb") as f:
                for chunk in img_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            manifest_rows.append({
                "species": folder_name,
                "obs_id": obs_id if obs_id is not None else "",
                "filename": filename.name,
                "phash": _compute_phash(filename),
            })

            downloaded += 1
            print(f"   [{downloaded}/{count}] {filename.name}")
            time.sleep(0.05)

        except Exception as exc:
            print(f"   [WARN] Skipped {url[:70]}  ({exc})")
            continue

    _append_manifest(manifest_path, manifest_rows)

    print(
        f"   Done: {downloaded} downloaded, {skipped} already existed -> {save_dir}"
    )
    return downloaded, skipped


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download bird images by species name using iNaturalist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "birds",
        nargs="*",
        metavar="BIRD",
        help="One or more bird species names (wrap multi-word names in quotes).",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="FILE",
        help="Path to a text file with one bird name per line. A line may also be "
             "'Search Name | folder_slug' to pin the output folder (keeps folder "
             "names aligned with classes.txt when iNaturalist uses a different name).",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=250,
        help="Number of images to download per species (default: 250).",
    )
    parser.add_argument(
        "--output", "-o",
        default="dataset",
        metavar="DIR",
        help="Root output directory (default: dataset/).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download images even if they already exist.",
    )
    return parser.parse_args()


def collect_bird_specs(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """
    Return a list of (search_name, folder_slug_or_None).
    A file/CLI entry of the form 'Search Name | folder_slug' pins the folder.
    """
    specs: list[tuple[str, str | None]] = []

    def add(entry: str) -> None:
        if "|" in entry:
            query, folder = entry.split("|", 1)
            specs.append((query.strip(), folder.strip()))
        else:
            specs.append((entry.strip(), None))

    for entry in args.birds:
        add(entry)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] Bird list file not found: {file_path}")
            sys.exit(1)
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    add(line)

    if not specs:
        print("[ERROR] No bird names provided. Use positional arguments or --file.")
        print('    Example: python download_bird_images.py "Laughing Kookaburra" --count 80')
        sys.exit(1)

    return specs


def main() -> None:
    args = parse_args()
    bird_specs = collect_bird_specs(args)
    output_dir = Path(args.output)
    skip_existing = not args.no_skip

    print("Bird Image Downloader  (source: iNaturalist)")
    print(f"    Species    : {len(bird_specs)}")
    print(f"    Per species: {args.count} images")
    print(f"    Output     : {output_dir.resolve()}")
    print(f"    Skip existing: {skip_existing}")
    print(f"    Manifest   : {(output_dir / MANIFEST_NAME).resolve()}")
    if not _HAVE_PHASH:
        print("    [WARN] 'imagehash' not installed - phash column will be empty.")
        print("           Install with: pip install imagehash  (needed for dedup_split.py)")

    total_dl = 0
    total_sk = 0
    empty_species: list[str] = []

    for query, folder in bird_specs:
        dl, sk = download_images(query, args.count, output_dir, skip_existing, folder)
        total_dl += dl
        total_sk += sk
        if dl == 0 and sk == 0:
            empty_species.append(query)

    print(f"\nAll done!  Total downloaded: {total_dl}  |  Skipped: {total_sk}")
    print(f"    Dataset saved to: {output_dir.resolve()}")
    if empty_species:
        print(f"\n[WARN] {len(empty_species)} species returned NO images "
              f"(check the search name):")
        for name in empty_species:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
