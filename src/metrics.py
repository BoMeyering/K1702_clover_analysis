"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""

import torch
from torchmetrics.classification import F1Score, JaccardIndex
from torchmetrics.segmentation import MeanIoU
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
import torch.distributed as dist

# Helper for DDP loss/metric reduction
def reduce_tensor(tensor):
    """ Reduce tensor across all GPUs """
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    return tensor

# Object Detection Metrics
class ObjectDetectionMetricLogger:
    def __init__(self, iou_threshold=0.5, box_format='xyxy', device='cpu'):
        self.device = device
        self.metrics = {
            "mAP": MeanAveragePrecision(iou_type="bbox").to(device).to(rank),
            "IoU": IntersectionOverUnion(iou_threshold=iou_threshold, box_format=box_format).to(device).to(rank),
        }

    def update(self, outputs, targets):
        """
        Args:
            outputs: list of dicts from Faster R-CNN ('boxes', 'scores', 'labels')
            targets: list of dicts with keys 'boxes', 'labels'
        """
        preds = []
        targets_gt = []

        for det in outputs:
            preds.append({
                "boxes": det["boxes"].detach().cpu(),
                "scores": det["scores"].detach().cpu(),
                "labels": det["labels"].detach().cpu()
            })

        for tgt in targets:
            targets_gt.append({
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu().to(torch.int64)
            })

        self.metrics["mAP"].update(preds, targets_gt)
        self.metrics["IoU"].update(preds, targets_gt)

    def compute(self):
        map_metrics = self.metrics["mAP"].compute()
        iou_metric = self.metrics["IoU"].compute()

        # reduce across GPUs if distributed
        flat_map_metrics = {f"mAP/{k}": reduce_tensor(v) for k, v in map_metrics.items()}
        iou_metric = reduce_tensor(iou_metric)

        return {**flat_map_metrics, "IoU": iou_metric}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

# Segmentation Metrics
class SegmentationMetricLogger:
    def __init__(self, num_classes: int, device: str):
        self.device = device
        # average metrics
        self.avg_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=False, input_format='index').to(device)
        }
        # per-class metrics
        self.mc_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass', average='none').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=True, input_format='index').to(device)
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.long().to(self.device)
        targets = targets.long().to(self.device)

        # update avg metrics
        for key, metric in self.avg_metrics.items():
            metric.update(preds, targets)
        # update per-class metrics
        for key, metric in self.mc_metrics.items():
            metric.update(preds, targets)

    def compute(self):
        avg_metrics = {k: reduce_tensor(metric.compute()) for k, metric in self.avg_metrics.items()}
        mc_metrics = {k: reduce_tensor(metric.compute()) for k, metric in self.mc_metrics.items()}
        return avg_metrics, mc_metrics

    def reset(self):
        for metric in self.avg_metrics.values():
            metric.reset()
        for metric in self.mc_metrics.values():
            metric.reset()

# Test
if __name__ == "__main__":
    # ---- Segmentation Example ----
    print("\n=== Segmentation Test ===")
    batches = 20
    num_classes = 5
    seg_metrics = SegmentationMetricLogger(num_classes=num_classes, device="cpu")

    for _ in range(batches):
        preds = torch.randn(10, num_classes, 20, 20)
        targets = torch.randint(0, num_classes, (10, 20, 20))
        preds_indices = torch.argmax(preds, dim=1)
        seg_metrics.update(preds=preds_indices, targets=targets)

    avg, mc = seg_metrics.compute()
    print("Average Metrics:", avg)
    print("Per-Class Metrics:", mc)

    # ---- Object Detection Example ----
    print("\n=== Object Detection Test ===")
    obj_metrics = ObjectDetectionMetricLogger(device="cpu")

    for _ in range(5):
        outputs = [{
            "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1])
        }]
        targets = [{
            "boxes": torch.tensor([[12, 12, 48, 48]], dtype=torch.float),
            "labels": torch.tensor([1])
        }]
        obj_metrics.update(outputs, targets)

    result = obj_metrics.compute()
    print("Detection Metrics:", result)
