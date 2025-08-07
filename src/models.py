"""
src/models.py
Models Stub script
BoMeyering 2025
"""

import torch
import inspect
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
from effdet import DetBenchTrain
from effdet import EfficientDet
from effdet import get_efficientdet_config
from effdet.efficientdet import HeadNet



def create_effdet_model(num_classes: int=3, image_size: tuple=(1024, 1024), architecture: str='efficientdet_d0', max_det_per_image: int=50, pretrained: bool=True):
    config = get_efficientdet_config(architecture)

    config.update({'num_classes': num_classes})
    config.update({'image_size': image_size})
    config.update({'max_det_per_image': max_det_per_image})

    net = EfficientDet(config, pretrained_backbone=pretrained)

    net.class_net = HeadNet(
        config=config,
        num_outputs=config.num_classes
    )

    return DetBenchTrain(net, config)


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

