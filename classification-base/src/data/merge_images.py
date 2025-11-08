#!/usr/bin/env python3
"""
Merge and deduplicate images from data/mac into data/mac-merged.

- Recursively walks data/mac for images that Pillow can open
- Deduplicates by hashing normalized pixel data (format-agnostic)
- Converts all unique images to PNG
- Saves as sequential filenames: 0.png, 1.png, ...
- Shows a progress bar while processing files

Usage:
    python merge_images.py [--src DIR] [--dst DIR]

Defaults:
    --src defaults to ./data/mac relative to this script
    --dst defaults to ./data/mac-merged relative to this script

Requirements:
    Pillow (pip install Pillow)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

from tqdm import tqdm

try:
    from PIL import Image, UnidentifiedImageError
except Exception as e:  # pragma: no cover
    print("This script requires Pillow. Install with: pip install Pillow", file=sys.stderr)
    raise


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"
}


def discover_files(root: Path) -> Iterable[Path]:
    """Yield all files under root recursively that look like images by extension."""
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # If it has an image-like extension, consider it. We'll still verify via PIL open.
        if p.suffix.lower() in IMAGE_EXTS or p.suffix:
            yield p


def normalized_image(img: Image.Image) -> Image.Image:
    """Return a normalized image for hashing/saving: convert to RGBA without metadata."""
    # Convert to RGBA to make hash robust across source formats/modes
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def image_fingerprint(img: Image.Image) -> str:
    """Compute a SHA-256 hash of the image's pixel data after normalization.

    Includes size and mode to avoid collisions between different sizes with identical byte streams.
    """
    norm = normalized_image(img)
    h = hashlib.sha256()
    # Include size and mode to strengthen fingerprint
    h.update(str(norm.size).encode("utf-8"))
    h.update(b"|")
    h.update(norm.mode.encode("utf-8"))
    h.update(b"|")
    h.update(norm.tobytes())
    return h.hexdigest()


def process_images(src_dir: Path, dst_dir: Path) -> Tuple[int, int, int]:
    """Process images: dedupe by content, convert to PNG, and enumerate names.

    Returns a tuple: (total_seen, unique_saved, skipped_corrupt)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    total_seen = 0
    unique_saved = 0
    skipped_corrupt = 0

    files = list(discover_files(src_dir))
    total_files = len(files)
    if total_files == 0:
        return 0, 0, 0

    with tqdm(total=total_files, desc="Processing images", unit="img") as pbar:
        for path in files:
            total_seen += 1
            try:
                with Image.open(path) as img:
                    # For animated images, take the first frame
                    try:
                        if getattr(img, "is_animated", False):
                            img.seek(0)
                    except Exception:
                        pass

                    fp = image_fingerprint(img)
                    if fp in seen_hashes:
                        # duplicate
                        continue
                    seen_hashes.add(fp)

                    norm = normalized_image(img)
                    out_name = f"{unique_saved}.png"
                    out_path = dst_dir / out_name
                    norm.save(out_path, format="PNG")
                    unique_saved += 1
            except (UnidentifiedImageError, OSError):
                # Unreadable/corrupt or unsupported
                skipped_corrupt += 1
            except Exception:
                # Unexpected issue: count as corrupt and continue
                skipped_corrupt += 1
            finally:
                duplicates = total_seen - unique_saved - skipped_corrupt
                try:
                    pbar.set_postfix(saved=unique_saved, duplicates=duplicates, corrupt=skipped_corrupt)
                except Exception:
                    # tqdm might fail on some terminals; ignore postfix errors
                    pass
                pbar.update(1)

    return total_seen, unique_saved, skipped_corrupt


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    default_src = (here / "data" / "laptops").resolve()
    default_dst = (here / "data" / "laptops-merged").resolve()

    ap = argparse.ArgumentParser(description="Merge, dedupe, and convert images to PNG.")
    ap.add_argument("--src", type=Path, default=default_src, help=f"Source dir (default: {default_src})")
    ap.add_argument("--dst", type=Path, default=default_dst, help=f"Destination dir (default: {default_dst})")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src_dir: Path = args.src
    dst_dir: Path = args.dst

    print(f"Source directory: {src_dir}")
    print(f"Destination directory: {dst_dir}")

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory does not exist or is not a directory: {src_dir}", file=sys.stderr)
        sys.exit(1)

    total, saved, skipped = process_images(src_dir, dst_dir)

    print("\nDone.")
    print(f"Total files discovered: {total}")
    print(f"Unique images saved:  {saved}")
    print(f"Skipped (corrupt/unsupported): {skipped}")


if __name__ == "__main__":
    main()
