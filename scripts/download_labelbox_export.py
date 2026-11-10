"""
scripts/download_labelbox_export.py
Export the finished (DONE) labels of a Labelbox project and download their instance masks.

Outputs (under --out-dir):
    export.ndjson                         raw export of the kept data rows, one per line
    masks/<img_id>/<nnn>_<class>.png      one binary (0/255) mask per segmentation instance
    masks/<img_id>/annotations.json       manifest: every object (mask file or bbox) with class + ids

Only data rows whose workflow status is in --statuses (default: DONE) are kept; the rest of the
project (IN_REWORK, TO_LABEL, ...) is discarded.

Requires the `labelbox` extra (`uv sync --extra labelbox`) and LABELBOX_API_KEY in the environment.

Usage:
    uv run scripts/download_labelbox_export.py
    uv run scripts/download_labelbox_export.py --project-id <id> --out-dir data/raw/labelbox_hq
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import labelbox
import numpy as np
import requests
from tqdm import tqdm

DEFAULT_PROJECT_ID = "cmfd63bm905sb07y97rdcgi8a"
DEFAULT_OUT_DIR = "data/raw/labelbox_hq"

# Include everything needed to rebuild masks: labels, metadata, attachments, image dims.
EXPORT_PARAMS = {
    "data_row_details": True,
    "metadata_fields": True,
    "attachments": True,
    "project_details": True,
    "label_details": True,
    "performance_details": False,
    "interpolated_frames": False,
}


def export_project(client: labelbox.Client, project_id: str) -> list[dict]:
    project = client.get_project(project_id)
    print(f'Exporting project "{project.name}" ({project_id})...')

    export_task = project.export(params=EXPORT_PARAMS)
    export_task.wait_till_done()

    if export_task.has_errors():
        errors = []
        export_task.get_buffered_stream(stream_type=labelbox.StreamType.ERRORS).start(
            stream_handler=lambda e: errors.append(e.json)
        )
        sys.exit(f"Export failed: {errors}")

    rows = []
    export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(
        stream_handler=lambda r: rows.append(r.json)
    )
    return rows


def fetch_mask(session: requests.Session, url: str, out_path: Path) -> tuple[int, int]:
    """Download one instance mask, store it as a single-channel 0/255 PNG, return its (h, w)."""
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    mask = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not decode mask from {url}")
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(str(out_path), mask)
    return mask.shape


def build_jobs(rows: list[dict], project_id: str, statuses: set[str], mask_dir: Path):
    """Write a manifest per image and return the (url, path) mask downloads it references."""
    jobs = []
    for row in rows:
        project = row["projects"][project_id]
        status = project["project_details"]["workflow_status"]
        if status not in statuses:
            continue
        if len(project["labels"]) != 1:
            print(
                f"WARNING: {row['data_row']['external_id']} has {len(project['labels'])} labels; using the first"
            )
        label = project["labels"][0]

        img_id = Path(row["data_row"]["external_id"]).stem
        img_dir = mask_dir / img_id
        img_dir.mkdir(parents=True, exist_ok=True)

        objects = []
        for i, obj in enumerate(label["annotations"]["objects"]):
            entry = {
                "name": obj["name"],
                "kind": obj["annotation_kind"],
                "feature_id": obj["feature_id"],
                "feature_schema_id": obj["feature_schema_id"],
            }
            if obj["annotation_kind"] == "ImageSegmentationMask":
                fname = f"{i:03d}_{obj['name']}.png"
                entry["mask_file"] = fname
                jobs.append((obj["mask"]["url"], img_dir / fname))
            elif obj["annotation_kind"] == "ImageBoundingBox":
                bb = obj["bounding_box"]
                entry["bbox_xyxy"] = [
                    bb["left"],
                    bb["top"],
                    bb["left"] + bb["width"],
                    bb["top"] + bb["height"],
                ]
            else:
                entry["raw"] = obj
            objects.append(entry)

        manifest = {
            "img_id": img_id,
            "data_row_id": row["data_row"]["id"],
            "label_id": label["id"],
            "workflow_status": status,
            "width": row["media_attributes"]["width"],
            "height": row["media_attributes"]["height"],
            "label_updated_at": label["label_details"]["updated_at"],
            "objects": objects,
        }
        with open(img_dir / "annotations.json", "w") as f:
            json.dump(manifest, f, indent=2)
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["DONE"],
        help="Workflow statuses to download masks for",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("LABELBOX_API_KEY")
    if not api_key:
        sys.exit("LABELBOX_API_KEY is not set")

    client = labelbox.Client(api_key=api_key)
    rows = export_project(client, args.project_id)

    statuses = set(args.statuses)
    n_total = len(rows)
    rows = [
        r
        for r in rows
        if r["projects"][args.project_id]["project_details"]["workflow_status"]
        in statuses
    ]
    print(f"Keeping {len(rows)} of {n_total} data rows (statuses: {sorted(statuses)})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.out_dir / "export.ndjson"
    with open(export_path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"Wrote {len(rows)} data rows to {export_path}")

    jobs = build_jobs(rows, args.project_id, statuses, args.out_dir / "masks")
    n_images = len({p.parent for _, p in jobs})
    print(
        f"Downloading {len(jobs)} instance masks for {n_images} images (statuses: {args.statuses})"
    )

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_mask, session, url, path) for url, path in jobs]
        for fut in tqdm(futures, desc="masks"):
            fut.result()

    print(f"Done. Masks in {args.out_dir / 'masks'}")


if __name__ == "__main__":
    main()
