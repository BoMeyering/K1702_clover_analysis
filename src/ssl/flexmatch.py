"""
src.ssl.flexmatch.py
FlexMatch curriculum pseudo-labelling for semantic segmentation.

FlexMatch (Zhang et al. 2021, https://arxiv.org/abs/2110.08263) gives every class its own confidence
threshold T_c = M(beta_c) * tau, where beta_c is the class's normalised "learning status" and M is a
mapping function. Here learning status is measured over pixels.

Ported from pgcview_v2_training_framework/src/flexmatch.py with these changes:
  * learning status can be tracked as a running EMA across steps (status='ema') instead of from the
    current batch alone (status='batch', the pgcview behaviour), and is all-reduced across DDP ranks
    so every rank uses the same thresholds;
  * normalize='ratio' measures status as (confident pixels of class c) / (pixels predicted as c), so a
    class that is rare by area (e.g. clover_stems, ~0.1% of pixels) is not mistaken for an unlearned
    class. normalize='count' is the paper's raw confident-count status;
  * the concave mapping is fixed (torch.log(2) on an int raised, which silently fell back to fixed tau).

BoMeyering 2025
"""

import math

import torch
import torch.distributed as dist

MAPPINGS = ("convex", "concave", "linear")


def map_beta(beta: torch.Tensor, mapping: str) -> torch.Tensor:
    """Apply the FlexMatch mapping function M to normalised learning status in [0, 1]."""
    if mapping == "convex":
        return beta / (2.0 - beta)
    if mapping == "concave":
        return torch.log1p(beta) / math.log(2.0)
    if mapping == "linear":
        return beta
    raise ValueError(f"mapping must be one of {MAPPINGS}, got '{mapping}'")


def class_beta(
    logits: torch.Tensor,
    tau: float = 0.85,
    mapping: str = "linear",
    warmup: bool = True,
) -> torch.Tensor:
    """Stateless, per-batch class beta (pgcview behaviour, raw confident counts). Returns mapped beta of shape (C,)."""
    if logits.ndim != 4:
        raise ValueError("Argument 'logits' should be a tensor of shape (N, C, H, W).")
    if not 0 < tau <= 1:
        raise ValueError("Argument 'tau' must be a float value in the interval (0, 1].")
    num_classes = logits.shape[1]
    max_probs, preds = torch.softmax(logits.float(), dim=1).max(dim=1)
    conf = max_probs > tau
    sigma = torch.bincount(preds[conf], minlength=num_classes).float()
    n_unused = (~conf).sum().float()
    denom = torch.maximum(sigma.max(), n_unused) if warmup else sigma.max()
    beta = sigma / denom.clamp_min(1.0)
    return map_beta(beta, mapping)


@torch.no_grad()
def get_pseudo_labels(
    tau_vector: torch.Tensor, logits_or_probs: torch.Tensor, probs: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard pseudo-labels and the boolean mask of pixels whose confidence clears their class threshold."""
    if logits_or_probs.ndim != 4:
        raise ValueError(f"Expected (N, C, H, W), got {tuple(logits_or_probs.shape)}")
    p = logits_or_probs if probs else torch.softmax(logits_or_probs.float(), dim=1)
    max_probs, pseudo_labels = p.max(dim=1)
    mask = max_probs.ge(tau_vector.to(p.device)[pseudo_labels])
    return pseudo_labels, mask


class FlexMatchThreshold:
    """Stateful class-wise thresholds for pixel-level FlexMatch."""

    def __init__(
        self,
        num_classes: int,
        tau: float = 0.95,
        mapping: str = "convex",
        warmup: bool = True,
        status: str = "ema",
        normalize: str = "ratio",
        momentum: float = 0.99,
        tau_min: float = 0.0,
        device: str = "cpu",
    ):
        if not 0 < tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if mapping not in MAPPINGS:
            raise ValueError(f"mapping must be one of {MAPPINGS}")
        if status not in ("ema", "batch"):
            raise ValueError("status must be 'ema' or 'batch'")
        if normalize not in ("ratio", "count"):
            raise ValueError("normalize must be 'ratio' or 'count'")

        self.num_classes = num_classes
        self.tau = tau
        self.tau_min = tau_min
        self.mapping = mapping
        self.warmup = warmup
        self.status = status
        self.normalize = normalize
        self.momentum = momentum
        self.device = device

        # Fractions of unlabeled pixels. FlexMatch starts with every sample "unused".
        self.conf_frac = torch.zeros(
            num_classes, device=device
        )  # confident & predicted as c
        self.pred_frac = torch.full(
            (num_classes,), 1.0 / num_classes, device=device
        )  # predicted as c
        self.unused_frac = torch.ones((), device=device)  # not confident
        self.steps = 0

    @torch.no_grad()
    def update(self, probs: torch.Tensor):
        """Update learning status from softmax probabilities of the weak view, shape (N, C, H, W)."""
        max_probs, preds = probs.max(dim=1)
        conf = max_probs > self.tau
        n = torch.tensor(float(preds.numel()), device=probs.device)
        stats = torch.cat(
            [
                torch.bincount(preds[conf], minlength=self.num_classes).float(),
                torch.bincount(preds.reshape(-1), minlength=self.num_classes).float(),
                (~conf).sum().float().reshape(1),
                n.reshape(1),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        C = self.num_classes
        total = stats[-1].clamp_min(1.0)
        conf_frac, pred_frac, unused_frac = (
            stats[:C] / total,
            stats[C : 2 * C] / total,
            stats[2 * C] / total,
        )

        if self.status == "batch":
            self.conf_frac, self.pred_frac, self.unused_frac = (
                conf_frac,
                pred_frac,
                unused_frac,
            )
        else:
            m = self.momentum
            self.conf_frac = m * self.conf_frac + (1 - m) * conf_frac
            self.pred_frac = m * self.pred_frac + (1 - m) * pred_frac
            self.unused_frac = m * self.unused_frac + (1 - m) * unused_frac
        self.steps += 1

    def learning_status(self) -> torch.Tensor:
        """Un-normalised learning status sigma_c."""
        if self.normalize == "count":
            return self.conf_frac
        return self.conf_frac / self.pred_frac.clamp_min(1e-12)

    def beta(self) -> torch.Tensor:
        """Normalised (pre-mapping) learning status in [0, 1]."""
        sigma = self.learning_status()
        denom = (
            torch.maximum(sigma.max(), self.unused_frac) if self.warmup else sigma.max()
        )
        return (sigma / denom.clamp_min(1e-12)).clamp(0.0, 1.0)

    def thresholds(self) -> torch.Tensor:
        """Class thresholds T_c = M(beta_c) * tau, floored at tau_min."""
        return (map_beta(self.beta(), self.mapping) * self.tau).clamp_min(self.tau_min)

    @torch.no_grad()
    def __call__(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the status with this batch, then pseudo-label it."""
        self.update(probs)
        return get_pseudo_labels(self.thresholds(), probs, probs=True)

    def state_dict(self) -> dict:
        return {
            "conf_frac": self.conf_frac.cpu(),
            "pred_frac": self.pred_frac.cpu(),
            "unused_frac": self.unused_frac.cpu(),
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict):
        self.conf_frac = state["conf_frac"].to(self.device)
        self.pred_frac = state["pred_frac"].to(self.device)
        self.unused_frac = state["unused_frac"].to(self.device)
        self.steps = state["steps"]


@torch.no_grad()
def cutmix_boxes(
    batch_size: int,
    height: int,
    width: int,
    p: float,
    device,
    area: tuple[float, float] = (0.02, 0.4),
    ratio: tuple[float, float] = (0.3, 1 / 0.3),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean (B, H, W) masks; True marks the box pasted from the partner sample. Rows are all False with prob 1-p."""
    boxes = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
    for i in range(batch_size):
        if torch.rand(1, generator=generator).item() > p:
            continue
        for _ in range(10):  # rejection-sample a box that fits
            a = (
                torch.empty(1).uniform_(*area, generator=generator).item()
                * height
                * width
            )
            r = math.exp(
                torch.empty(1)
                .uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=generator)
                .item()
            )
            h, w = round(math.sqrt(a / r)), round(math.sqrt(a * r))
            if 0 < h <= height and 0 < w <= width:
                y = torch.randint(0, height - h + 1, (1,), generator=generator).item()
                x = torch.randint(0, width - w + 1, (1,), generator=generator).item()
                boxes[i, y : y + h, x : x + w] = True
                break
    return boxes


@torch.no_grad()
def apply_cutmix(
    images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, boxes: torch.Tensor
):
    """Paste each sample's box region from its partner (the batch rolled by one) into images, labels and masks."""
    if not boxes.any() or images.shape[0] < 2:
        return images, labels, masks
    partner = torch.roll(torch.arange(images.shape[0], device=images.device), shifts=1)
    b3 = boxes.unsqueeze(1)
    images = torch.where(b3, images[partner], images)
    labels = torch.where(boxes, labels[partner], labels)
    masks = torch.where(boxes, masks[partner], masks)
    return images, labels, masks
