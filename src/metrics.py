"""
src.metrics.py
Torchmetrics for image prediction
BoMeyering 2025
"""
import torch
from torchmetrics.classification import F1Score, Accuracy, JaccardIndex, Precision, Recall
from torchmetrics.segmentation import MeanIoU
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection import MeanAveragePrecision

class ObjectDetectionMetricLogger:
    def __init__(self, iou_threshold=0.5, box_format='xyxy', device='cpu'):
        """
        Initializes the object detection metrics.

        Args:
            iou_threshold (float): IoU threshold for true positive determination.
            box_format (str): Format of bounding boxes ('xyxy' or 'cxcywh').
            device (str): Device to run metrics on.
        """
        self.metrics = {
            'mAP': MeanAveragePrecision(iou_type='bbox').to(device),
            'IoU': IntersectionOverUnion(iou_threshold=iou_threshold, box_format=box_format).to(device)
        }

    def update(self, outputs, targets):
        """
        Processes Faster R-CNN outputs and updates metrics.

        Args:
            outputs (List[Dict]): List of detections from Faster R-CNN, 
                                each dict has 'boxes', 'scores', 'labels'.
            targets (Dict): Raw targets containing 'bbox' and 'cls'.
        """
        preds = []
        for det in outputs:  # already a list of dicts
            preds.append({
                "boxes": det["boxes"].detach().cpu(),
                "scores": det["scores"].detach().cpu(),
                "labels": det["labels"].detach().cpu()
            })

        targets_gt = []
        for i in range(len(targets['bbox'])):
            targets_gt.append({
                "boxes": targets['bbox'][i].detach().cpu(),
                "labels": targets['cls'][i].detach().cpu().to(torch.int64)
            })

        # Update metrics
        self.metrics['mAP'].update(preds, targets_gt)
        self.metrics['IoU'].update(preds, targets_gt)


    def compute(self):
        map_metrics = self.metrics['mAP'].compute()
        iou_metric = self.metrics['IoU'].compute()

        # Flatten mAP dictionary and prefix keys (the return )
        flat_map_metrics = {f"mAP/{k}": v for k, v in map_metrics.items()}

        return {
            **flat_map_metrics,
            "IoU": iou_metric
        }

    def reset(self):
        self.metrics['mAP'].reset()
        self.metrics['IoU'].reset()


class SegmentationMetricLogger:
    """ Class to log metrics during an epoch """
    def __init__(self, num_classes: int, device: str):
        """
        Initialize the MetricLogger

        Args:
            num_classes (int): total number of classes to track
            device (str): Where the computation will be taking place
        """
        self.avg_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=False, input_format='index').to(device)
            # 'accuracy': Accuracy(num_classes=num_classes, task='multiclass').to(device),
            # 'precision': Precision(num_classes=num_classes, task='multiclass').to(device),
            # 'recall': Recall(num_classes=num_classes, task='multiclass').to(device)
        }
        self.mc_metrics = {
            'f1_score': F1Score(num_classes=num_classes, task='multiclass', average='none').to(device),
            'jaccard_index': JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device),
            'mIOU': MeanIoU(num_classes=num_classes, include_background=True, per_class=True, input_format='index').to(device)
            # 'accuracy': Accuracy(num_classes=num_classes, task='multiclass', average='none').to(device),
            # 'precision': Precision(num_classes=num_classes, task='multiclass', average='none').to(device),
            # 'recall': Recall(num_classes=num_classes, task='multiclass', average='none').to(device)
        }
        self.batch_results = {
            'avg': {},
            'mc': {}
        }
    
    def update(self, preds: torch.tensor, targets: torch.tensor, verbose: bool=False):
        # Ensure preds and targets are long tensors (class indices)
        preds = preds.long()
        targets = targets.long()

        # update avg metrics
        for key, metric in self.avg_metrics.items():
            self.batch_results['avg'][key] = metric(preds, targets)
        
        # update multiclass metrics
        for key, metric in self.mc_metrics.items():
            self.batch_results['mc'][key] = metric(preds, targets)
        
        if verbose:
            self.print_metrics('both')
        
    def compute(self):
        try:
            avg_metrics = {k: metric.compute() for k, metric in self.avg_metrics.items()}
            mc_metrics = {k: metric.compute() for k, metric in self.mc_metrics.items()}
        except Exception as e:
            print(e)
            avg_metrics, mc_metrics = None, None
        return avg_metrics, mc_metrics
    
    def print_metrics(self, type: str):
        if type=='avg':
            print(self.batch_results['avg'])
        elif type=='mc':
            print(self.batch_results['mc'])
        elif type=='both':
            print(self.batch_results)

    def reset(self):
        for k, metric in self.avg_metrics.items():
            metric.reset()
        for k, metric in self.mc_metrics.items():
            metric.reset()


if __name__ == '__main__':
    batches = 20
    num_classes = 5
    metrics = SegmentationMetricLogger(num_classes, device='cpu')         

    for i in range(batches):
        preds = torch.randn(10, 5, 20, 20)
        targets = torch.randint(num_classes, (10, 20, 20))
        # targets = torch.argmax(preds.softmax(dim=1), dim=1)
        preds_indices = torch.argmax(preds, dim=1)  # shape (10, 20, 20) # very important because we want to make sure we send the correct format definetly do this before the call to function


        metrics.update(preds=preds_indices, targets=targets, verbose=False)

    avg, mc = metrics.compute()

    print(avg)
    print(mc)

    # metrics.reset()
    # avg, mc = metrics.compute()

    # print(avg)
    # print(mc)
    
