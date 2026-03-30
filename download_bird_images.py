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
import hashlib
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests


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


# ─────────────────────────────────────────────
#  iNaturalist photo URL fetcher
# ─────────────────────────────────────────────

def _inaturalist_photo_urls(taxon_name: str, max_results: int) -> list[str]:
    """
    Fetch photo URLs from iNaturalist research-grade observations.
    Returns medium-sized image URLs (typically 800px wide).
    """
    urls: list[str] = []
    page = 1

    while len(urls) < max_results:
        need = max_results - len(urls)
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

        try:
            resp = requests.get(INAT_API, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"iNaturalist API error: {exc}") from exc

        results = data.get("results", [])
        if not results:
            break  # no more pages

        for obs in results:
            for photo in obs.get("photos", []):
                raw_url = photo.get("url", "")
                if not raw_url:
                    continue
                # Swap square thumbnail for medium (better quality for training)
                med_url = re.sub(r"/square\.", "/medium.", raw_url)
                med_url = re.sub(r"/square\b", "/medium", med_url)
                if med_url not in urls:
                    urls.append(med_url)
                if len(urls) >= max_results:
                    break
            if len(urls) >= max_results:
                break

        total = data.get("total_results", 0)
        if page * per_page >= total:
            break  # exhausted all pages

        page += 1
        time.sleep(0.5)  # be polite to iNaturalist

    return urls[:max_results]


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


# ─────────────────────────────────────────────
#  Main downloader
# ─────────────────────────────────────────────

def download_images(
    bird_name: str,
    count: int,
    output_dir: Path,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """
    Download `count` images for `bird_name` into output_dir/<folder>/.
    Returns (downloaded, skipped) counts.
    """
    folder_name = _safe_folder_name(bird_name)
    save_dir = output_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching: '{bird_name}' -> {save_dir}")

    try:
        urls = _inaturalist_photo_urls(bird_name, max_results=count + 30)
    except Exception as exc:
        print(f"   [ERROR] Failed to fetch URLs: {exc}")
        return 0, 0

    if not urls:
        print("   [WARN] No photos found on iNaturalist for this species.")
        return 0, 0

    downloaded = 0
    skipped = 0

    for url in urls:
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

            downloaded += 1
            print(f"   [{downloaded}/{count}] {filename.name}")
            time.sleep(0.05)

        except Exception as exc:
            print(f"   [WARN] Skipped {url[:70]}  ({exc})")
            continue

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
        help="Path to a text file with one bird name per line.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=80,
        help="Number of images to download per species (default: 80).",
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


def collect_bird_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = list(args.birds)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[ERROR] Bird list file not found: {file_path}")
            sys.exit(1)
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    names.append(name)

    if not names:
        print("[ERROR] No bird names provided. Use positional arguments or --file.")
        print('    Example: python download_bird_images.py "Laughing Kookaburra" --count 80')
        sys.exit(1)

    return names


def main() -> None:
    args = parse_args()
    bird_names = collect_bird_names(args)
    output_dir = Path(args.output)
    skip_existing = not args.no_skip

    print("Bird Image Downloader  (source: iNaturalist)")
    print(f"    Species    : {len(bird_names)}")
    print(f"    Per species: {args.count} images")
    print(f"    Output     : {output_dir.resolve()}")
    print(f"    Skip existing: {skip_existing}")

    total_dl = 0
    total_sk = 0

    for bird in bird_names:
        dl, sk = download_images(bird, args.count, output_dir, skip_existing)
        total_dl += dl
        total_sk += sk

    print(f"\nAll done!  Total downloaded: {total_dl}  |  Skipped: {total_sk}")
    print(f"    Dataset saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
