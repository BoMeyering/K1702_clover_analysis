"""
src/utils/postprocess.py
Detection and Segmentation Post Processing Utilities
BoMeyering 2025
"""

import cv2
import scipy
import numpy as np
from typing import Iterable
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

#----------------------------------------------------------------------------------#
# Bounding Box Formatting
#----------------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------------#
# ROI Transformations
#----------------------------------------------------------------------------------#
def get_corner_pts(img_shape: Iterable) -> np.ndarray:
    """Get the corner points of an image

    Parameters:
    -----------
        img_shape : Iterable[int, int]
            A tuple of integers in the format (H, W)

    Raises:
    -------
        ValueError: 
            If 'img_shape' is not an Iterable type of length 2
            Or if elements are not integers.

    Returns:
    --------
        img_corners : np.ndarray
            An array of shape (4, 2) that contains the corner points for an image
            Sorted in clockwise fashion starting at the top left corner (0, 0) 
    """
    if not isinstance(img_shape, Iterable) or len(img_shape) != 2:
        raise ValueError(
            f"'img_shape' should be an Iterable of 2 integers; got a length {len(img_shape)} {type(img_shape)} instead."
        )
    if not all([isinstance(i, int) for i in img_shape]):
        raise ValueError(
            f"'img_shape' should be an iterable of 2 integers; check element type."
        )
    
    img_corners = np.array(
        [
            [0, 0],
            [img_shape[1], 0],
            [img_shape[1], img_shape[0]],
            [0, img_shape[0]]
        ],
        dtype=np.int64
    )

    return img_corners

def order_pts(pts: np.ndarray, img_shape: Iterable) -> np.ndarray:
    """Order a set of 4 points in clockwise orientation

    Parameters:
    -----------
        pts : array-like
            A array of shape (4, 2) of (possibly) unordered detections in x,y format

    Returns:
    --------
        idx : np.ndarray
            An array of shape (4,) of the row index of pts sorted from top left in clockwise fashion
    """
    try:
        pts = np.asarray(pts, dtype=np.float64)
    except Exception as e:
        raise ValueError(
            f"Problem coercing pts to np.ndarray with dtype ``np.float64``. Error: {e}"
        )
    if pts.shape != (4, 2):
        raise ValueError(
            f"'pts' should be an array of shape (4, 2); got {pts.shape} instead"
        )
    if not np.isfinite(pts).all():
        raise ValueError(
            "'pts' contains non-finite values"
        )

    img_corners = get_corner_pts(img_shape)
    
    # Calculate the Euclidean Cost matrix and optimize
    dst_M = distance_matrix(img_corners, pts)
    idx = linear_sum_assignment(dst_M)

    return idx[1]

def point_transform(img: np.ndarray, pts: np.ndarray, output_shape: Iterable) -> np.ndarray:
    """Perform a 4 point transform on an image ROI

    Parameters:
    -----------
        img : array-like
            A single or three channel array-like image in the format (H, W, C).
        pts : array-like
            An array of shape (4, 2) of ordered corner points in (x, y) format.
        output_shape : Iterable
            An iterable of the desired output shape of the image ROI.

    Returns:
    --------
        warped : np.ndarray
            A new image of the transformed ROI whose corners correspond to an image of shape (output_shape).
    """

    output_corners = get_corner_pts(output_shape).astype(np.float32)

    M = cv2.getPerspectiveTransform(pts, output_corners)
    warped = cv2.warpPerspective(img, M, output_shape)

    return warped

#----------------------------------------------------------------------------------#
# Prediction Resizing
#----------------------------------------------------------------------------------#

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

#----------------------------------------------------------------------------------#
# Drawing Functions
#----------------------------------------------------------------------------------#

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

# def draw_bounding_boxes(img: np.ndarray, bboxes: np.ndarray, bbox_color_map, scale_factor=0.005, padding_factor=0.005, font_thickness_factor=0.001) -> 

if __name__ == '__main__':
    pts = np.array(
        [
            [700, 430],
            [1200, 1200],
            [50, 200],
            [350, 700]
        ]
    )
    idx = order_pts(pts, img_shape=(1500, 2000))

    ordered_pts = pts[idx, ].astype(np.float32)
    print(ordered_pts)

    img = np.random.randint(0, 255, (1500, 2000, 3)).astype(np.float32)

    warped = point_transform(img, ordered_pts, output_shape=(256, 256)).astype(np.uint8)

    def show_image(img):
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    show_image(img.astype(np.uint8))
    show_image(warped)