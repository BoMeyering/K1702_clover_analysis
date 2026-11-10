"""
scripts/build_sainfoin_mat_targets.py
Rasterize the sainfoin leaf instance segmentation masks into one semantic class-index
.mat file per image, matching the K1702 SAM mask format (src/mask_generation/src/inference.py):
a single 2D array under the 'data' key.

Reads   data/raw/sainfoin_leaves/labelbox_hq/masks/<img_id>/{annotations.json, NNN_<class>.png}
Writes  data/processed_hq/sainfoin_leaves/targets/<img_id>.mat   (uint8, 'data' key)
        data/processed_hq/sainfoin_leaves/targets/<img_id>.png   (same class-index array; read by k1702 train)

Classes (metadata/sainfoin_leaf_class_map.json): 0 background, 1 leaflet, 2 petiole, 3 tag.
Pixels not covered by any instance mask stay background (0); overlapping instances
resolve last-instance-wins, same as build_hq_targets.py.

Usage:
    uv run scripts/build_sainfoin_mat_targets.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat


def build_target(img_dir: Path, class_map: dict) -> tuple[np.ndarray, Counter, int]:
    manifest = json.loads((img_dir / "annotations.json").read_text())
    target = np.zeros((manifest["height"], manifest["width"]), dtype=np.uint8)
    painted = np.zeros(target.shape, dtype=np.uint8)

    for obj in manifest["objects"]:
        if "mask_file" not in obj:
            continue
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
        "--mask-dir",
        type=Path,
        default=Path("data/raw/sainfoin_leaves/labelbox_hq/masks"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed_hq/sainfoin_leaves/targets"),
    )
    parser.add_argument(
        "--class-map", type=Path, default=Path("metadata/sainfoin_leaf_class_map.json")
    )
    args = parser.parse_args()

    class_map = json.loads(args.class_map.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_dirs = sorted(
        p for p in args.mask_dir.iterdir() if (p / "annotations.json").exists()
    )
    total = Counter()
    for img_dir in img_dirs:
        target, counts, n_overlap = build_target(img_dir, class_map)
        savemat(
            args.out_dir / f"{img_dir.name}.mat",
            {"data": target, "project": "sainfoin_leaves"},
        )
        if not cv2.imwrite(str(args.out_dir / f"{img_dir.name}.png"), target):
            raise OSError(f"Could not write {args.out_dir / img_dir.name}.png")
        total += counts
        warn = (
            f"  WARNING: {n_overlap} overlapping px (last instance wins)"
            if n_overlap
            else ""
        )
        print(
            f"{img_dir.name}: background {100 * counts.get(0, 0) / target.size:.2f}%{warn}"
        )

    names = {v: k for k, v in class_map.items()}
    n = sum(total.values())
    print(f"Wrote {len(img_dirs)} targets to {args.out_dir}")
    for k, v in sorted(total.items()):
        print(f"  {k:>3} {names.get(k, '?'):<10} {100 * v / n:6.2f}%")


if __name__ == "__main__":
    main()
