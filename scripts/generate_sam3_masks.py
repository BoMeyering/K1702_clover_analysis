"""
scripts/generate_sam3_masks.py
Regenerate the point-prompted segmentation masks with SAM3 (or SAM 2.1) for comparison with the
original SAM v1 masks in data/processed/targets.

Prompting reproduces src/mask_generation/src/inference.py (the SAM v1 run):
  * one prompt per class, all of that class's points as positive clicks, multimask_output=True,
    keep the highest-scoring mask;
  * classes are painted in priority order quadrat -> soil -> clover; a pixel keeps the first class
    that claims it;
  * output encoding matches data/processed/targets: 0 = soil or unclaimed, 1 = quadrat, 2 = clover.
Differences from the v1 run: images are passed as RGB (v1 read them as BGR because
cv2.COLOR_BGR2RGB was passed as the imread flag), and the point dates are corrected (below).

Point -> image mapping: every date in data/SAM_points.csv is swapped. Points labelled 20170608 belong
to the 20170703 image and vice versa (verified against the v1 masks: 99% point/mask agreement swapped,
51% as labelled). The two 'unkown' accessions have no image and are skipped.

Outputs (under --out-dir, default data/SAM3_mask):
  masks/<img_id>.png          class-index mask, full resolution, same encoding as data/processed/targets
  overlays/<img_id>.jpg       1/4-scale panel: image + points | SAM v1 mask | new mask
  summary.csv                 per image: point counts, chosen mask scores, unclaimed fraction,
                              point agreement and IoU vs the SAM v1 mask per class
  SAM_points_corrected.csv    the point table with corrected dates / img_ids
  run_info.json               model and options used

Requires transformers >= 5 and, for SAM3, access to the gated facebook/sam3 repo on Hugging Face.
Without touching the project environment:
    uv run --with "transformers>=5.22" scripts/generate_sam3_masks.py
    uv run --with "transformers>=5.22" scripts/generate_sam3_masks.py --model facebook/sam2.1-hiera-large --limit 5
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

CLASS_ORDER = ["quadrat", "soil", "clover"]  # paint priority (same as the v1 run)
OUT_CODE = {"soil": 0, "quadrat": 1, "clover": 2}  # data/processed/targets encoding
PALETTE = np.array(
    [[0, 0, 0], [255, 0, 200], [40, 220, 40]], dtype=np.uint8
)  # RGB per OUT_CODE; soil is left untinted
POINT_BGR = {"soil": (40, 60, 160), "quadrat": (255, 0, 0), "clover": (0, 255, 0)}


def corrected_points(path: Path, image_dir: Path) -> pl.DataFrame:
    """Swap the (mislabelled) capture date of every point and keep points whose image exists."""
    pts = pl.read_csv(path)
    swapped = pl.when(pl.col("date") == 20170608).then(20170703).otherwise(20170608)
    pts = (
        pts.with_columns(
            original_img_id=pl.col("img_id"),
            date=swapped,
        )
        .with_columns(
            img_id=pl.col("accession") + "_" + pl.col("date").cast(pl.Utf8),
        )
        .with_columns(img_name=pl.col("img_id") + ".jpg")
    )
    have = {p.stem for p in image_dir.glob("*.jpg")}
    return pts.filter(pl.col("img_id").is_in(list(have)))


def load_backend(model_id: str, device: str):
    import transformers

    if "sam3" in model_id.lower():
        model_cls, proc_cls = (
            transformers.Sam3TrackerModel,
            transformers.Sam3TrackerProcessor,
        )
    elif "sam2" in model_id.lower():
        model_cls, proc_cls = transformers.Sam2Model, transformers.Sam2Processor
    else:
        raise ValueError(
            f"Unsupported model {model_id}: expected a SAM3 or SAM2 checkpoint"
        )
    processor = proc_cls.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id).to(device).eval()
    return model, processor


@torch.no_grad()
def segment_image(
    model,
    processor,
    img_rgb: np.ndarray,
    pts: pl.DataFrame,
    device: str,
    negatives: bool,
    multimask: bool,
):
    """Return (class mask uint8, unclaimed bool mask, {class: score})."""
    h, w = img_rgb.shape[:2]
    enc = processor(images=img_rgb, return_tensors="pt").to(device)
    embeddings = model.get_image_embeddings(enc["pixel_values"])
    original_sizes = enc["original_sizes"]

    mask = np.zeros((h, w), dtype=np.uint8)
    claimed = np.zeros((h, w), dtype=bool)
    scores = {}
    for cls in CLASS_ORDER:
        pos = pts.filter(pl.col("class") == cls).select("x", "y").to_numpy()
        if len(pos) == 0:
            continue
        coords, labels = pos.tolist(), [1] * len(pos)
        if negatives:
            neg = pts.filter(pl.col("class") != cls).select("x", "y").to_numpy()
            coords += neg.tolist()
            labels += [0] * len(neg)

        prompt = processor(
            input_points=[[coords]],
            input_labels=[[labels]],
            original_sizes=original_sizes,
            return_tensors="pt",
        ).to(device)
        out = model(
            input_points=prompt["input_points"],
            input_labels=prompt["input_labels"],
            image_embeddings=embeddings,
            multimask_output=multimask,
        )
        iou = out.iou_scores[0, 0].float()  # (num_masks,)
        best = int(torch.argmax(iou))
        masks = processor.post_process_masks(
            out.pred_masks[:, :, best : best + 1], original_sizes
        )[0]
        cls_mask = masks[0, 0].cpu().numpy().astype(bool)
        scores[cls] = float(iou[best])

        new = cls_mask & ~claimed
        mask[new] = OUT_CODE[cls]
        claimed |= cls_mask
    return mask, ~claimed, scores


def compare(mask: np.ndarray, ref: np.ndarray, pts: pl.DataFrame) -> dict:
    row = {}
    for cls, code in OUT_CODE.items():
        a, b = mask == code, ref == code
        union = np.logical_or(a, b).sum()
        row[f"iou_v1_{cls}"] = (
            float(np.logical_and(a, b).sum() / union) if union else float("nan")
        )
    xs = np.clip(pts["x"].to_numpy().astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(pts["y"].to_numpy().astype(int), 0, mask.shape[0] - 1)
    want = np.array([OUT_CODE[c] for c in pts["class"]])
    row["point_agreement_new"] = float((mask[ys, xs] == want).mean())
    row["point_agreement_v1"] = float((ref[ys, xs] == want).mean())
    return row


def overlay(img_rgb, ref, mask, pts, scale=4, alpha=0.45):
    small = cv2.resize(
        img_rgb,
        (img_rgb.shape[1] // scale, img_rgb.shape[0] // scale),
        interpolation=cv2.INTER_AREA,
    )
    size = (small.shape[1], small.shape[0])

    def blend(m):
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        out = small.copy()
        tint = m > 0
        out[tint] = (alpha * PALETTE[m[tint]] + (1 - alpha) * small[tint]).astype(
            np.uint8
        )
        return out

    base = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    for cls, x, y in pts.select("class", "x", "y").iter_rows():
        cv2.circle(base, (int(x) // scale, int(y) // scale), 6, POINT_BGR[cls], -1)
        cv2.circle(base, (int(x) // scale, int(y) // scale), 6, (0, 0, 0), 1)
    panels = [base]
    for title, m in (("SAM v1", ref), ("new", mask)):
        p = cv2.cvtColor(blend(m) if m is not None else small, cv2.COLOR_RGB2BGR)
        cv2.putText(p, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
        cv2.putText(
            p, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2
        )
        panels.append(p)
    return np.concatenate(panels, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default="facebook/sam3", help="facebook/sam3 or a SAM2 checkpoint"
    )
    parser.add_argument("--points", type=Path, default=Path("data/SAM_points.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/processed/images"))
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("data/processed/targets"),
        help="SAM v1 masks to compare against",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/SAM3_mask"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids", nargs="*", help="Only these img_ids")
    parser.add_argument("--limit", type=int, help="Only the first N images (sorted)")
    parser.add_argument(
        "--negatives",
        action="store_true",
        help="Add the other classes' points as negative clicks",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="multimask_output=False (one mask per prompt)",
    )
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo images whose mask already exists"
    )
    args = parser.parse_args()

    for sub in ("masks", "overlays"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    pts = corrected_points(args.points, args.image_dir)
    pts.write_csv(args.out_dir / "SAM_points_corrected.csv")
    img_ids = sorted(pts["img_id"].unique().to_list())
    if args.ids:
        img_ids = [i for i in img_ids if i in set(args.ids)]
    if args.limit:
        img_ids = img_ids[: args.limit]

    (args.out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "negatives": args.negatives,
                "multimask_output": not args.single_mask,
                "class_order": CLASS_ORDER,
                "encoding": OUT_CODE,
                "n_images": len(img_ids),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    model, processor = load_backend(args.model, args.device)
    print(
        f"Loaded {args.model} on {args.device}; {len(img_ids)} images -> {args.out_dir}"
    )

    summary_path = args.out_dir / "summary.csv"
    done = set()
    if summary_path.exists() and not args.overwrite:
        done = {r["img_id"] for r in csv.DictReader(open(summary_path))}
    fieldnames = [
        "img_id",
        "n_quadrat",
        "n_soil",
        "n_clover",
        "score_quadrat",
        "score_soil",
        "score_clover",
        "unclaimed_frac",
        "iou_v1_soil",
        "iou_v1_quadrat",
        "iou_v1_clover",
        "point_agreement_new",
        "point_agreement_v1",
        "seconds",
    ]
    new_file = not summary_path.exists() or args.overwrite
    with open(summary_path, "w" if new_file else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for img_id in tqdm(img_ids):
            out_mask = args.out_dir / "masks" / f"{img_id}.png"
            if img_id in done and out_mask.exists():
                continue
            t0 = time.time()
            img_pts = pts.filter(pl.col("img_id") == img_id)
            img = cv2.cvtColor(
                cv2.imread(str(args.image_dir / f"{img_id}.jpg")), cv2.COLOR_BGR2RGB
            )
            mask, unclaimed, scores = segment_image(
                model,
                processor,
                img,
                img_pts,
                args.device,
                args.negatives,
                not args.single_mask,
            )
            cv2.imwrite(str(out_mask), mask)

            ref_path = args.ref_dir / f"{img_id}.png"
            ref = (
                cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
                if ref_path.exists()
                else None
            )
            row = {"img_id": img_id, "unclaimed_frac": float(unclaimed.mean())}
            for cls in OUT_CODE:
                row[f"n_{cls}"] = int((img_pts["class"] == cls).sum())
                row[f"score_{cls}"] = scores.get(cls, float("nan"))
            if ref is not None:
                row.update(compare(mask, ref, img_pts))
            if not args.no_overlays:
                cv2.imwrite(
                    str(args.out_dir / "overlays" / f"{img_id}.jpg"),
                    overlay(img, ref, mask, img_pts),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
            row["seconds"] = round(time.time() - t0, 2)
            writer.writerow(row)
            f.flush()

    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
