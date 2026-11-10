"""
src.ssl.metrics.py
Ported from pgcview_v2_training_framework/src/metrics.py.

Torchmetrics for image prediction
BoMeyering 2025
"""

import logging

import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

logger = logging.getLogger()


class MetricLogger:
    """Wrapper for torchmetrics.MetricCollection"""

    def __init__(self, name: str, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Parameters:
        -----------
            num_classes : int
                The total number of classes to track
            device : torch.device
                The computational device for metric calculation
        """
        self.name = name
        self.avg_metrics = MetricCollection(
            [
                MeanIoU(
                    num_classes=num_classes, per_class=False, input_format="index"
                ).to(device),
                GeneralizedDiceScore(
                    num_classes=num_classes, per_class=False, input_format="index"
                ).to(device),
                # HausdorffDistance(num_classes=num_classes, input_format='index').to(device)
            ]
        )

        self.mc_metrics = MetricCollection(
            [
                MeanIoU(
                    num_classes=num_classes, per_class=True, input_format="index"
                ).to(device),
                GeneralizedDiceScore(
                    num_classes=num_classes, per_class=True, input_format="index"
                ).to(device),
            ]
        )

        self.results = {"avg": {}, "mc": {}}

    def update(self, preds: torch.tensor, targets: torch.tensor, verbose: bool = False):
        # update avg metrics
        self.avg_metrics.update(preds, targets)

        # update multiclass metrics
        self.mc_metrics.update(preds, targets)

    def compute(self):
        try:
            self.results["avg"] = self.avg_metrics.compute()
            self.results["mc"] = self.mc_metrics.compute()
        except Exception as e:
            logger.error(f"Encountered error when computing metrics. Error: {e}")
            self.results["avg"], self.results["mc"] = None, None

    def __str__(self) -> str:
        """
        Nicely format the metric results for printing.
        """
        lines = [self.name + ":"]

        def _format_dict(name: str, d: dict):
            if d is None or len(d) == 0:
                return f"   {name}: None"
            out = [f"   {name}:"]
            for k, v in d.items():
                # handle tensor outputs from torchmetrics
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = f"{v.item():.5f}"
                    else:
                        v = [round(x, 5) for x in v.flatten().tolist()]
                if isinstance(v, float):
                    out.append(f"       {k:<20s}: {v:.4f}")
                else:
                    out.append(f"       {k:<20s}: {v}")
            return "\n".join(out)

        lines.append(_format_dict("Average", self.results.get("avg")))
        lines.append(_format_dict("Multi-class", self.results.get("mc")))

        return "\n" + "\n".join(lines)

    def reset(self):
        """Rest Metric Collections"""
        self.avg_metrics.reset()
        self.mc_metrics.reset()
