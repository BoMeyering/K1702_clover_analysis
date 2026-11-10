"""
scripts/build_hq_targets.py
Rasterize the high-quality Labelbox instance masks into one semantic class-index PNG per image.

Reads   data/raw/labelbox_hq/masks/<img_id>/{annotations.json, NNN_<class>.png}
Writes  data/processed_hq/targets/<img_id>.png   (uint8, values from the class map; 255 = unlabeled)

These targets are kept separate from the old SAM-derived masks in data/processed/targets.
The target mapping (default metadata/hq_target_mapping.json, {name: {class_idx, rgb}}) maps Labelbox
class names to indices; several names may share a class_idx to merge classes. Unmapped names are an error.

Usage:
    uv run scripts/build_hq_targets.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IGNORE_INDEX = 255


def build_target(img_dir: Path, class_map: dict) -> tuple[np.ndarray, Counter, int]:
    manifest = json.loads((img_dir / "annotations.json").read_text())
    target = np.full(
        (manifest["height"], manifest["width"]), IGNORE_INDEX, dtype=np.uint8
    )
    painted = np.zeros(target.shape, dtype=np.uint8)

    for obj in manifest["objects"]:
        if "mask_file" not in obj:
            continue  # bounding boxes (quadrat_corner) are not part of the segmentation target
        if obj["name"] not in class_map:
            raise KeyError(
                f"{img_dir.name}: class '{obj['name']}' is not in the class map"
            )
        inst = cv2.imread(str(img_dir / obj["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
        target[inst] = class_map[obj["name"]]
        painted += inst

    counts = Counter(
        {int(k): int(v) for k, v in zip(*np.unique(target, return_counts=True))}
    )
    n_overlap = int((painted > 1).sum())
    return target, counts, n_overlap


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mask-dir", type=Path, default=Path("data/raw/labelbox_hq/masks")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed_hq/targets")
    )
    parser.add_argument(
        "--target-mapping", type=Path, default=Path("metadata/hq_target_mapping.json")
    )
    args = parser.parse_args()

    class_map = {
        name: v["class_idx"]
        for name, v in json.loads(args.target_mapping.read_text()).items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_dirs = sorted(
        p for p in args.mask_dir.iterdir() if (p / "annotations.json").exists()
    )
    total = Counter()
    for img_dir in img_dirs:
        target, counts, n_overlap = build_target(img_dir, class_map)
        cv2.imwrite(str(args.out_dir / f"{img_dir.name}.png"), target)
        total += counts
        warn = (
            f"  WARNING: {n_overlap} overlapping px (last instance wins)"
            if n_overlap
            else ""
        )
        print(
            f"{img_dir.name}: unlabeled {100 * counts[IGNORE_INDEX] / target.size:.2f}%{warn}"
        )

    names = {v: k for k, v in class_map.items()}
    n = sum(total.values())
    print(f"Wrote {len(img_dirs)} targets to {args.out_dir}")
    for k, v in sorted(total.items()):
        print(f"  {k:>3} {names.get(k, 'ignore'):<14} {100 * v / n:6.2f}%")


if __name__ == "__main__":
    main()
