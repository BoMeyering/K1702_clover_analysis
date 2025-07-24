"""
Models Stub script
BoMeyering 2025
"""

import torch
import inspect
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation



def create_model(num_classes: int, variant: str='nvidia/mit-b0'):
    variants = [
        'nvidia/mit-b0', 
        'nvidia/mit-b1', 
        'nvidia/mit-b2', 
        'nvidia/mit-b3', 
        'nvidia/mit-b4', 
        'nvidia/mit-b5'
    ]
    if variant in variants:
        model = SegformerForSemanticSegmentation.from_pretrained(
            variant,
            num_labels=num_classes,
        )
        return model
    else:
        raise ValueError(f"Arg 'variant' must be one of {variants}")
    

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

