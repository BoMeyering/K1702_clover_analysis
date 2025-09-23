"""
src/models.py
Models Stub script
BoMeyering 2025
"""

import torch
import torchvision
import inspect
import numpy as np
import segmentation_models_pytorch as smp
from typing import Tuple, Union
from collections import OrderedDict
from torch import argmax
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterrcnn_model(architecture: str, num_classes: int, max_det_per_image: int, pretrained: bool=True, **kwargs):
    """Create a pytorch FasterRCNN model

    Parameters:
    -----------
        architecture : str

        num_classes : int

        max_det_per_image : int
            
        pretrained : bool, optional
            _description_. Defaults to True.

    Raises:
    -------
        ValueError: _description_

    Returns:
    --------
        model : torch.nn.Module
            An instantiated 
    """
    if architecture == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Replace head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Limit max detections per image
        model.roi_heads.detections_per_img = max_det_per_image

        return model
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

def create_smp_model(config: dict) -> torch.nn.Module:
    """Creates an smp Pytorch model

    conf:
        conf (omegaconf.dictconfig.DictConfig): The OmegaConf configuration dictionary

    Raises:
        ValueError: If conf.model.config.encoder_name is not listed in smp.encoders.get_encoder_names().
        ValueError: If conf.model.architecture does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """
    
    if config['encoder_name'] not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {config['encoder_name']} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    
    try:
        model_class = getattr(smp, config['architecture'])
        class_arguments = [name for name, param in inspect.signature(model_class).parameters.items()]
        model_args = {}
        for k, v in config.items():
            if k in class_arguments:
                model_args[k] = v
        model = model_class(**model_args)

        return model
    except AttributeError as e:
        raise ValueError(f"Model architecture {config['architecture']} is not a valid SMP architecture.\nSelect one from 'smp._MODEL_ARCHITECTURES'")
    

def load_models(config, device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load and return both models

    Parameters:
    -----------
        config : Omegaconf.conf
            Loaded inference config yaml file as Omegaconf
        device : torch.device
            Computational device to load the models on to

    Returns:
    --------
        models : Tuple[torch.nn.Module, torch.nn.Module]
            The segmentation model and the object detection model
    """
    seg_model, det_model = None, None

    # ----------------- Segmentation -----------------
    if getattr(config, "enable_segmentation", False):
        seg_model = create_smp_model(config=config.segmentation.model_config)
        checkpoint = torch.load(config.segmentation.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)

        new_state = OrderedDict()
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v

        seg_model.load_state_dict(new_state, strict=False)
        seg_model.to(device).eval()
        print("[info] Loaded segmentation model")

    # ----------------- Detection -----------------
    if getattr(config, "enable_detection", False):
        det_cfg = {
            "architecture": config.detection.get("architecture", "fasterrcnn_resnet50_fpn"),
            "pretrained": config.detection.get("pretrained", True),
            "num_classes": int(config.detection.get("num_classes", 2)),
            "max_det_per_image": int(config.detection.get("max_det_per_image", 20)),
            "image_size": tuple(config.detection.get("input_size", (1024, 1024)))
        }

        det_model = create_fasterrcnn_model(**det_cfg)
        checkpoint = torch.load(config.detection.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        new_state = OrderedDict()
        for k, v in state_dict.items():
            nk = k.replace("module.", "").replace("model.", "")
            new_state[nk] = v

        det_model.load_state_dict(new_state, strict=False)
        det_model.to(device).eval()
        print("[info] Loaded detection model (Faster R-CNN)")

    return seg_model, det_model

@torch.no_grad()
def run_segmentation_inference(
    model: torch.nn.Module, 
    img_tensor: torch.Tensor, 
    device: Union[torch.device, str]
    ) -> np.ndarray:
    """Run segmentation inference

    Run semantic segmentation inference on a 3 channel transformed tensor image.

    Parameters:
    -----------
        model : torch.nn.Module
            An instantiated segmentation model
        img_tensor : torch.Tensor
            A 3 channel torch tensor image with shape (C, H, W)
        device : Union[torch.device, str]
            The computational device used for inference

    Returns:
    --------
        preds : np.ndarray
            The prediction map of shape (H, W) where each pixel is the predicted class.
    """
    # Add batch dimension
    if img_tensor.ndim != 3:
        raise ValueError(
            f"'img_tensor' should be a torch.tensor with shape (C, H, W); got shape {img_tensor.shape} instead."
        )
    else:
        img_tensor = img_tensor.unsqueeze(0).to(device)

    # Calculate the predictions
    logits = model(img_tensor)
    preds = argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return preds

@torch.no_grad()
def run_detection_inference(
    model: torch.nn.Module, 
    img_tensor: torch.Tensor,
    device: Union[torch.device, str] 
    ) -> np.ndarray:
    """Run object detection inference

    Run object detection inference on a 3 channel transformed tensor image.

    Parameters:
    -----------
        model : torch.nn.Module
            An instantiated segmentation model
        img_tensor : torch.Tensor
            A 3 channel torch tensor image with shape (C, H, W)
        device : Union[torch.device, str]
            The computational device used for inference

    Returns:
    --------
        preds : np.ndarray
            The predicted bounding boxes from the model
    """

    # Add batch dimension
    # Add batch dimension
    if img_tensor.ndim != 3:
        raise ValueError(
            f"'img_tensor' should be a torch.tensor with shape (C, H, W); got shape {img_tensor.shape} instead."
        )
    else:
        img_tensor = img_tensor.unsqueeze(0).to(device)
    
    outputs = model(img_tensor)[0]
    outputs = format_FRCNN_output(outputs)

    return outputs

def format_FRCNN_output(detections: dict) -> np.ndarray:
    """Format the FasterRCNN outputs

    Reformat the Faster RCNN output dictionary to a numpy array

    Parameters:
    -----------
        detections : dict
            A dictionary of torch.tensors with keys
            'boxes', 'scores', 'labels'
            boxes tensor should be of shape (N, 4)
            scores tensor should be of shape (N,)
            labels tensor should be of shape (N,)

    Returns:
    --------
        detection_arr : np.ndarray
            An np.ndarray of the predicted bounding boxes, scores, and class labels of shape (N, 6)
    """

    boxes = detections['boxes'].detach().cpu().numpy()
    scores = detections['scores'].detach().cpu().numpy()
    labels = detections['labels'].detach().cpu().numpy()

    shape_set = set([boxes.shape[0], scores.shape[0], labels.shape[0]])
    if len(shape_set) != 1:
        raise ValueError(
            f"The lenght of the first dimension of 'boxes', 'labels', and 'scores' differ. "\
            "Please ensure that the predictions returns consistent data"
        )
    
    detection_arr = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis])).astype(np.float32)

    return detection_arr



if __name__ == '__main__':
    detections = {
        'boxes': torch.Tensor([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [4, 5, 6, 7]
        ]),
        'scores': torch.Tensor([.3454, .9999, .945, .921]),
        'labels': torch.Tensor([1, 4, 4, 4])
    }

    detections = format_obj_output(detections=detections)

    print(detections)