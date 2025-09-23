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

def resize_map(preds: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize prediction maps

    Resizes a predicted segmentation map of shape (H_p, W_p) to (H_o, W_o)
    Where H_p, W_p are the prediction map height and width and
    H_o, W_o are the output height and width

    Parameters:
    -----------
        preds : np.ndarray
            An np.ndarray of shape (H_p, W_p) with the predicted classes of the image
        output_shape : Tuple[int, int]
            A tuple of integers of the output image shape (H_o, W_o)

    Returns:
    --------
        preds : np.ndarray
            An np.ndarray of the resized prediction map
    """
    
    try:
        preds = np.asarray(preds, dtype=np.uint8)
    except Exception as e:
        raise ValueError(
            f"Encountered error casting preds to type ``np.uint8``. Check data integrity. "\
            f"Error: {e}"
        )
    if len(output_shape) != 2:
        raise ValueError(
            f"'output_shape' should be a tuple of two integers; got a length {len(output_shape)} {type(output_shape)} instead."
        )
    # Convert output_shape to (W, H) to convert to OpenCV standards
    output_shape = tuple(output_shape[::-1])
    preds = cv2.resize(src=preds, dsize=output_shape)

    return preds

def resize_boxes(detections: np.ndarray, pred_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize detections

    Resizes predicted bounding boxes from (H_p, W_p) to (H_o, W_o)
    Where H_p, W_p are the detections height and width and
    H_o, W_o are the output height and width

    Parameters:
    -----------
        detections : np.ndarray
            An np.ndarray of shape (N, 6) with the predicted bounding boxes, scores, and labels
            Boxes (columns 0-3) are in (x1, y1, x2, y2) format
            Scores (column 4) are floats in the interval [0, 1]
            Labels (column 5) are integers in the interval [1, num_classes]
        prediction_shape : Tuple[int, int]
            A tuple of integers of the prediction image shape (H_p, W_p)
        output_shape : Tuple[int, int]
            A tuple of integers of the output image shape (H_o, W_o) to scale the bounding boxes to

    Returns:
    --------
        preds : np.ndarray
            An np.ndarray of the resized prediction map
    """
    # Set scale factors
    scale_x = output_shape[1] / pred_shape[1]
    scale_y = output_shape[0] / pred_shape[0]

    # Resize boxes
    detections[:, [0, 2]] *= scale_x
    detections[:, [1, 3]] *= scale_y

    return detections

def draw_overlay(
        img: np.ndarray, 
        preds: np.ndarray, 
        color_map: np.ndarray, 
        alpha=0.5, beta=0.5, 
        gamma=0.1
    ) -> np.ndarray:
    """Overlay prediction map

    Overlay a prediction map of shape (H, W) onto an image of shape (H, W)

    Parameters:
    -----------
        img : np.ndarray
            _description_
        preds : np.ndarray
            _description_
        color_map : np.ndarray
            _description_
        alpha : float, optional
            _description_. Defaults to 0.5.
        beta : float, optional
            _description_. Defaults to 0.5.
        gamma : float, optional
            _description_. Defaults to 0.1.

    Returns:
    --------
        overlay : np.ndarray
            _description_
    """

    color_mask = color_map[preds].astype(np.uint8)
    overlay = cv2.addWeighted(img.copy(), alpha, color_mask, beta, gamma)

    return overlay

def draw_bounding_boxes(img: np.ndarray, bboxes: np.ndarray, bbox_color_map, scale_factor=0.005, padding_factor=0.005, font_thickness_factor=0.001) -> 

if __name__ == '__main__':
    detections = {
        'boxes': torch.Tensor([
            [200, 200, 250, 250],
            [800, 250, 850, 300],
            [800, 850, 850, 900],
            [200, 800, 250, 850]
        ]),
        'scores': torch.Tensor([.3454, .9999, .945, .921]),
        'labels': torch.Tensor([1, 4, 4, 4])
    }

    detections = format_FRCNN_output(detections=detections)

    detections = resize_boxes(detections=detections, pred_shape=(1000, 1000), output_shape=(1500, 2500))

    print(detections)


    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, center=[100, 100], radius=50, color=255, thickness=-1)

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

    mask = resize_map(preds=mask, output_shape=(1500, 2500))

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()