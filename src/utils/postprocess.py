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
# Drawing functions
#----------------------------------------------------------------------------------#

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