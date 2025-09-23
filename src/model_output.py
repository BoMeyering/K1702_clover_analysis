"""
src/model_output.py
Run inference and format outputs
BoMeyering 2025
"""
import torch
import cv2
import numpy as np
from typing import Union, Tuple
from torch import argmax

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





if __name__ == '__main__':
    pass