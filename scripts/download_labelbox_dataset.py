"""
scripts/download_labelbox_dataset.py
Download every image in a named Labelbox *dataset* (not a project export) to
data/raw/<dataset-name>/images/, plus a manifest.csv of data row metadata.

This is for pulling raw, unlabeled assets straight off a Labelbox dataset. For
pulling finished annotations off a Labelbox *project*, see
scripts/download_labelbox_export.py instead.

Requires the `labelbox` extra (`uv sync --extra labelbox`) and LABELBOX_API_KEY
in the environment.

Usage:
    uv run scripts/download_labelbox_dataset.py --dataset-name sainfoin_leaves
    uv run scripts/download_labelbox_dataset.py --dataset-name sainfoin_leaves --out-dir data/raw/sainfoin_leaves --workers 16
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import labelbox as lb
import requests
from tqdm import tqdm

DEFAULT_OUT_ROOT = Path('data/raw')


def get_dataset(client: lb.Client, name: str) -> lb.Dataset:
    datasets = list(client.get_datasets(where=lb.Dataset.name == name))
    if not datasets:
        sys.exit(f'No Labelbox dataset found named "{name}"')
    if len(datasets) > 1:
        sys.exit(f'Found {len(datasets)} Labelbox datasets named "{name}"; expected exactly 1')
    return datasets[0]


def fetch_image(session: requests.Session, url: str, out_path: Path) -> None:
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-name', required=True, help='Exact Labelbox dataset name')
    parser.add_argument('--out-dir', type=Path, default=None,
                         help='Output dir (default: data/raw/<dataset-name>)')
    parser.add_argument('--workers', type=int, default=16, help='Parallel download threads')
    args = parser.parse_args()

    out_dir = args.out_dir or (DEFAULT_OUT_ROOT / args.dataset_name)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    client = lb.Client(api_key=os.environ['LABELBOX_API_KEY'])
    dataset = get_dataset(client, args.dataset_name)
    print(f'Dataset "{dataset.name}" ({dataset.uid}): {dataset.row_count} rows')

    rows = list(dataset.data_rows())

    manifest_rows = []
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for row in rows:
            filename = Path(row.external_id).name
            out_path = img_dir / filename
            manifest_rows.append({
                'data_row_id': row.uid,
                'global_key': row.global_key,
                'external_id': row.external_id,
                'filename': filename,
            })
            futures[pool.submit(fetch_image, session, row.row_data, out_path)] = filename

        for fut in tqdm(as_completed(futures), total=len(futures), desc='Downloading'):
            filename = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f'FAILED {filename}: {e}')

    manifest_path = out_dir / 'manifest.csv'
    with manifest_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['data_row_id', 'global_key', 'external_id', 'filename'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'Wrote {len(manifest_rows)} images to {img_dir}')
    print(f'Wrote manifest to {manifest_path}')


if __name__ == '__main__':
    main()
