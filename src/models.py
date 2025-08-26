"""
src/models.py
Models Stub script
BoMeyering 2025
"""

import torch
import torchvision
import inspect
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterrcnn_model(architecture, pretrained, num_classes, max_det_per_image, **kwargs):
    if architecture == "fasterrcnn_resnet50_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Replace head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Limit max detections per image
        model.roi_heads.detections_per_img = max_det_per_image

        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")



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