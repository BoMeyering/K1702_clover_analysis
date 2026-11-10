"""
src/predict.py
Batch segmentation inference with a src/ssl checkpoint; every output is written at the input image's
full resolution.

The model sees the whole image resized to the training size (config images.resize), as in training and
validation. Its logits are upsampled bilinearly to the original image size on the GPU before the argmax,
which gives smoother class boundaries than upsampling a low-resolution label map.

Outputs under --output-dir:
    masks/<stem>.png      class-index mask (uint8, values are class_idx)
    color/<stem>.png      mask rendered with the target-mapping colours
    overlay/<stem>.jpg    colour mask alpha-blended over the photo (skipped classes left untouched)
    class_areas.csv       per-image pixel count and fraction for every class
BoMeyering 2025
"""

import csv
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_KINDS = ("mask", "color", "overlay")
REPO_ROOT = Path(__file__).resolve().parents[1]

# Fallback colours when the checkpoint's target mapping cannot be found
DEFAULT_PALETTE = [
    (0, 0, 0),
    (40, 200, 40),
    (255, 0, 255),
    (0, 120, 255),
    (255, 200, 0),
    (120, 85, 50),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 255),
    (128, 128, 128),
]


# ----------------------------------------------------------------------------------#
# Model loading
# ----------------------------------------------------------------------------------#
def _architecture_name(arch) -> str:
    """The checkpoint stores a ModelArchitecture enum (or its name / value as a string)."""
    if hasattr(arch, "value"):
        return arch.value
    from src.ssl.config import ModelArchitecture

    return (
        ModelArchitecture[arch].value
        if arch in ModelArchitecture.__members__
        else str(arch)
    )


def _class_colors(conf: dict, class_names: list[str]) -> np.ndarray:
    """RGB colour per class index from the checkpoint's target mapping, falling back to a fixed palette."""
    path = conf.get("metadata", {}).get("target_mapping_path")
    candidates = [Path(path), REPO_ROOT / path] if path else []
    for p in candidates:
        if p.is_file():
            mapping = json.loads(p.read_text())
            colors = np.zeros((len(class_names), 3), dtype=np.uint8)
            for v in mapping.values():
                if v["class_idx"] < len(class_names):
                    colors[v["class_idx"]] = v["rgb"]
            return colors
    logger.warning("Target mapping %s not found; using the default palette", path)
    return np.array(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(len(class_names))],
        dtype=np.uint8,
    )


def load_checkpoint(path: str, weights: str = "ema", device: str = "cuda"):
    """Rebuild the model from the config saved inside a src/ssl checkpoint.

    Returns (model, conf dict, class names, class colours).
    """
    chkpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in chkpt:
        raise ValueError(
            f"{path} has no embedded config; expected a checkpoint written by src/ssl (k1702 train)"
        )
    conf = chkpt["config"]
    key = f"{weights}_state_dict"
    if key not in chkpt:
        raise KeyError(
            f"{path} has no {key} (available: {[k for k in chkpt if k.endswith('_state_dict')]})"
        )

    model_cfg = dict(conf["model"]["config"])
    model_cfg["encoder_weights"] = (
        None  # weights come from the checkpoint; skip the ImageNet download
    )
    model = getattr(smp, _architecture_name(conf["model"]["architecture"]))(**model_cfg)
    model.load_state_dict({k.removeprefix("module."): v for k, v in chkpt[key].items()})
    model.to(device).eval()

    n_classes = model_cfg["classes"]
    class_names = list(
        chkpt.get("class_names") or [f"class_{i}" for i in range(n_classes)]
    )
    return model, conf, class_names, _class_colors(conf, class_names)


# ----------------------------------------------------------------------------------#
# Data
# ----------------------------------------------------------------------------------#
def collect_images(inputs: Sequence[str], recursive: bool = False) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated list of image paths."""
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            paths.extend(
                f for f in it if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Input {inp} does not exist")
    return sorted(set(paths))


class PredictDataset(Dataset):
    """Full-resolution RGB image plus the normalised, model-sized tensor."""

    def __init__(
        self,
        paths: Sequence[Path],
        resize: tuple[int, int],
        means: Sequence[float],
        std: Sequence[float],
    ):
        self.paths = list(paths)
        self.resize = tuple(resize)  # (H, W)
        self.means = np.array(means, dtype=np.float32) * 255
        self.std = np.array(std, dtype=np.float32) * 255

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        bgr = cv2.imread(
            str(path), cv2.IMREAD_COLOR
        )  # applies EXIF orientation, as in training
        if bgr is None:
            return {"path": str(path), "error": "unreadable"}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # INTER_AREA approximates training, which decoded JPEGs at reduced size before resizing
        small = cv2.resize(
            rgb, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA
        )
        x = (small.astype(np.float32) - self.means) / self.std
        return {
            "path": str(path),
            "image": torch.from_numpy(rgb),
            "input": torch.from_numpy(x.transpose(2, 0, 1).copy()),
        }


def _collate(batch):
    ok = [b for b in batch if "error" not in b]
    return {
        "bad": [b["path"] for b in batch if "error" in b],
        "paths": [b["path"] for b in ok],
        "images": [b["image"] for b in ok],
        "input": torch.stack([b["input"] for b in ok]) if ok else None,
    }


# ----------------------------------------------------------------------------------#
# Inference
# ----------------------------------------------------------------------------------#
@torch.no_grad()
def predict_probs(
    model: torch.nn.Module, x: torch.Tensor, tta: bool = False, amp: bool = True
) -> torch.Tensor:
    """Softmax probabilities at model resolution (B, C, h, w), optionally averaged over flip TTA."""
    use_amp = amp and x.is_cuda
    with torch.autocast(
        device_type=x.device.type, dtype=torch.bfloat16, enabled=use_amp
    ):
        probs = model(x).float().softmax(1)
        if tta:
            for dims in ([3], [2]):  # horizontal, vertical flip
                probs += model(x.flip(dims)).float().softmax(1).flip(dims)
            probs /= 3
    return probs


@torch.no_grad()
def render(
    image: torch.Tensor,
    probs: torch.Tensor,
    colors: torch.Tensor,
    alpha: float,
    skip: torch.Tensor,
    kinds: Sequence[str],
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Upsample one image's probabilities to its full resolution and draw the requested outputs.

    image: (H, W, 3) uint8 on device; probs: (C, h, w). Returns the full-res mask and BGR / grey arrays to write.
    """
    H, W = image.shape[:2]
    mask = (
        F.interpolate(probs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        .argmax(0)
        .to(torch.uint8)
    )
    out = {}
    if "mask" in kinds:
        out["mask"] = mask.cpu().numpy()
    if "color" in kinds or "overlay" in kinds:
        color = colors[mask.long()]  # (H, W, 3) RGB
        if "color" in kinds:
            out["color"] = color.flip(-1).cpu().numpy()
        if "overlay" in kinds:
            blend = (
                (image.float() * (1 - alpha) + color.float() * alpha)
                .round()
                .to(torch.uint8)
            )
            keep = torch.isin(mask, skip)[..., None]
            out["overlay"] = torch.where(keep, image, blend).flip(-1).cpu().numpy()
    return mask, out


def _write(path: Path, arr: np.ndarray, params: list | None = None):
    if not cv2.imwrite(str(path), np.ascontiguousarray(arr), params or []):
        raise OSError(f"Failed to write {path}")


def run_predict(
    checkpoint: str,
    inputs: Sequence[str],
    output_dir: str,
    weights: str = "ema",
    device: str | None = None,
    save: Sequence[str] = OUTPUT_KINDS,
    alpha: float = 0.5,
    overlay_skip: Sequence[int] = (0,),
    tta: bool = False,
    amp: bool = True,
    batch_size: int = 2,
    num_workers: int = 4,
    recursive: bool = False,
    jpeg_quality: int = 95,
    resize: tuple[int, int] | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, conf, class_names, colors_np = load_checkpoint(
        checkpoint, weights=weights, device=device
    )
    resize = tuple(resize or conf["images"]["resize"])
    norm = conf.get("metadata", {}).get("norm", {})
    logger.info(
        "Loaded %s (%s weights, %s classes: %s) on %s; model input %s",
        checkpoint,
        weights,
        len(class_names),
        class_names,
        device,
        resize,
    )

    paths = collect_images(inputs, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"No images found in {list(inputs)}")
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(
            "Input images have duplicate file stems; outputs would overwrite each other"
        )
    logger.info("Predicting %d images -> %s", len(paths), output_dir)

    out_root = Path(output_dir)
    for kind in save:
        (out_root / {"mask": "masks"}.get(kind, kind)).mkdir(
            parents=True, exist_ok=True
        )

    ds = PredictDataset(
        paths,
        resize,
        norm.get("means", (0.485, 0.456, 0.406)),
        norm.get("std", (0.229, 0.224, 0.225)),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=device.startswith("cuda"),
    )
    colors = torch.from_numpy(colors_np).to(device)
    skip = torch.tensor(list(overlay_skip), dtype=torch.uint8, device=device)
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    rows, pending = [], []
    try:
        from tqdm import tqdm

        progress = tqdm(total=len(paths), unit="img")
    except ImportError:
        progress = None

    with ThreadPoolExecutor(
        max_workers=4
    ) as pool:  # PNG/JPEG encoding of full-res images overlaps with the GPU
        for batch in loader:
            for bad in batch["bad"]:
                logger.warning("Could not read %s; skipped", bad)
            if batch["input"] is None:
                continue
            probs = predict_probs(
                model, batch["input"].to(device, non_blocking=True), tta=tta, amp=amp
            )
            for path, image, p in zip(batch["paths"], batch["images"], probs):
                stem = Path(path).stem
                mask, out = render(
                    image.to(device, non_blocking=True), p, colors, alpha, skip, save
                )
                counts = (
                    torch.bincount(mask.flatten().long(), minlength=len(class_names))
                    .cpu()
                    .tolist()
                )
                total = mask.numel()
                row = {"image": path, "height": mask.shape[0], "width": mask.shape[1]}
                for name, c in zip(class_names, counts):
                    row[f"{name}_px"] = c
                    row[f"{name}_frac"] = round(c / total, 6)
                rows.append(row)

                if "mask" in out:
                    pending.append(
                        pool.submit(
                            _write, out_root / "masks" / f"{stem}.png", out["mask"]
                        )
                    )
                if "color" in out:
                    pending.append(
                        pool.submit(
                            _write, out_root / "color" / f"{stem}.png", out["color"]
                        )
                    )
                if "overlay" in out:
                    pending.append(
                        pool.submit(
                            _write,
                            out_root / "overlay" / f"{stem}.jpg",
                            out["overlay"],
                            jpeg,
                        )
                    )
            # Bound the number of full-res arrays waiting to be written
            while len(pending) > 16:
                pending.pop(0).result()
            if progress:
                progress.update(len(batch["paths"]) + len(batch["bad"]))
        for f in pending:
            f.result()
    if progress:
        progress.close()

    csv_path = out_root / "class_areas.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()) if rows else ["image"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d predictions and %s", len(rows), csv_path)
    return rows
