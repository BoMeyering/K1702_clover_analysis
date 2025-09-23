"""
inference.py
Model inference and post-processing script
Mirza Sakiroglu and BoMeyering 2025
"""

import os
import glob
import cv2
import numpy as np
import torch
from glob import glob
from omegaconf import OmegaConf
from collections import OrderedDict
from torch import argmax
from typing import Tuple, Union
from src.models import create_smp_model, create_fasterrcnn_model, load_models
from src.transforms import get_inf_transforms
from src.utils.postprocess import get_corner_pts, point_transform, order_pts



# ----------------- ROI Extractor -----------------
class ROIExtractor:
    def __init__(self, output_dir: str, output_shape=(512, 512)):
        """
        Args:
            output_dir (str): Where to save cropped masked ROI images
            output_shape (tuple): Desired output (H, W) for the perspective transform
        """
        # New folder for masked ROI crops
        self.output_dir = "outputs/cropped_masked_images"
        self.output_shape = output_shape
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_from_detections(self, image: np.ndarray, detections: list, image_path: str, overlay_img: np.ndarray) -> str:
        """
        Extracts ROI if exactly 4 quadrat corners are detected and saves only the masked version.
        """
        if len(detections) != 4:
            print(f"[info] Skipping ROI extraction: need 4 corners, got {len(detections)}")
            return None

        pts = np.array([
            [(x1 + x2) / 2, (y1 + y2) / 2] for (_, x1, y1, x2, y2, _) in detections
        ], dtype=np.float32)

        try:
            # ROI with binary segmentation mask only
            roi_overlay = point_transform(overlay_img, pts, output_shape=self.output_shape)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            roi_overlay_out_path = os.path.join(self.output_dir, f"{base_name}_masked.png")
            cv2.imwrite(roi_overlay_out_path, roi_overlay)
            print(f"[info] Saved ROI with clover mask: {roi_overlay_out_path}")

            return roi_overlay_out_path
        except Exception as e:
            print(f"[warn] Failed ROI transform: {e}")
            return None


# ----------------- Main -----------------
def main():
    config = OmegaConf.load("inference_config.yaml")

    color_map = np.array(config.color_map, dtype=np.uint8)
    bbox_color_map = {k: tuple(v) for k, v in config.bbox_color_map.items()}

    # Transforms
    image_size = tuple(config.get('image_resize', (1024, 1024)))
    inf_transforms = get_inf_transforms(resize=image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    seg_model, det_model = load_models(config, device)
    seg_model.eval()
    det_model.eval()

    # ROI extractor instance
    roi_extractor = ROIExtractor(config.output_dir, output_shape=(512, 512))

    # Grab all image names from input_folder
    image_paths = [file for file in glob('*', root_dir=config.input_folder) if file.endswith(('png', 'jpg'))]

    if not image_paths:
        print(f"[warn] No images found in {config.input_folder}")
        return

    for img_path in image_paths:
        process_image(
            img_path,
            seg_model,
            det_model,
            inf_transforms,
            color_map,
            bbox_color_map,
            config,
            device,
            roi_extractor
        )


# ----------------- Segmentation inference -----------------



# ----------------- Detection inference -----------------
@torch.no_grad()
def run_detection_inference(
    model: torch.nn.Module, 
    orig_img 
    obj_transforms, 
    device, 
    score_thresh=0.0, 
    label_map=None
    ) -> np.ndarray:
    orig_h, orig_w = orig_img.shape[:2]

    sample = obj_transforms(image=orig_img, bboxes=[], labels=[])
    image_tensor = sample["image"].to(device)

    resized_h, resized_w = image_tensor.shape[1:]

    with torch.no_grad():
        outputs = model([image_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        print(outputs)

        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes = boxes.astype(int)

        if label_map is None:
            label_map = {i: f"class_{i}" for i in range(1, model.num_classes)}

        results = []
        for (x1, y1, x2, y2), cid, sc in zip(boxes, labels, scores):
            if sc >= score_thresh:
                label_str = label_map.get(cid, f"class_{cid}")
                results.append((label_str, int(x1), int(y1), int(x2), int(y2), float(sc)))

    return results


# ----------------- Drawing -----------------
def draw_segmentation_overlay(image, preds, color_map, alpha=0.5, beta=0.5, gamma=0.1):
    color_mask = color_map[preds].astype(np.uint8)
    if color_mask.shape[:2] != image.shape[:2]:
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(image.copy(), alpha, color_mask, beta, gamma)
    return overlay


def draw_bounding_boxes(image, bboxes, bbox_color_map, scale_factor=0.005, padding_factor=0.005, font_thickness_factor=0.001):
    h = image.shape[0]
    for item in bboxes:
        if len(item) >= 6:
            label, x1, y1, x2, y2, score = item
        else:
            label, x1, y1, x2, y2 = item
            score = None

        color = bbox_color_map.get(label, (0, 255, 0))
        thickness = max(1, int(h * scale_factor))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        font_scale = 1.0
        thickness_text = max(int(h * font_thickness_factor), 1)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)
        padding = int(padding_factor * h)

        rx_start, ry_start = x1, max(0, y1 - text_height - 2 * padding)
        rx_end, ry_end = x1 + text_width + 2 * padding, y1
        cv2.rectangle(image, (rx_start, ry_start), (rx_end, ry_end), (0, 0, 0), cv2.FILLED)

        text_pos = (x1 + padding, y1 - padding)
        cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

    return image


# ----------------- Process image -----------------
def process_image(
        image_path, 
        seg_model, 
        det_model, 
        inf_transforms,
        color_map, 
        bbox_color_map, 
        config, device, 
        roi_extractor
    ):
    """
    """

    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    if orig_img is None:
        print(f"[warn] can't read {image_path}")
        return None

    combined = orig_img.copy()
    seg_mask = None

    if seg_model is not None:
        sample = seg_transforms(image=orig_img, mask=None)
        img_tensor = sample["image"]
        seg_mask = run_segmentation_inference(seg_model, img_tensor, device)
        combined = draw_segmentation_overlay(combined, seg_mask, color_map)

    detections = []
    if det_model is not None:
        detections = run_detection_inference(
            det_model,
            orig_img,
            obj_transforms=obj_transforms,
            device=device,
            score_thresh=getattr(config.detection, "score_thresh", 0.7),
            label_map={1: "quadrat_corner"}
        )

        if len(detections) == 0:
            print(f"[info] No detections found for {image_path}")
        else:
            print(f"[info] Detections for {image_path}:")
            for det in detections:
                label, x1, y1, x2, y2, score = det
                print(f"   {label} | box=({x1},{y1},{x2},{y2}) | score={score:.3f}")

        combined_with_boxes = draw_bounding_boxes(combined.copy(), detections, bbox_color_map)
    else:
        combined_with_boxes = combined.copy()

    # ---- ROI Extraction with clover-only mask ----
    if detections and seg_mask is not None:
        # Isolate clover class (adjust class ID if needed)
        CLOVER_CLASS_ID = 2
        clover_mask = (seg_mask == CLOVER_CLASS_ID).astype(np.uint8) * 255

        if clover_mask.shape != orig_img.shape[:2]:
            clover_mask = cv2.resize(clover_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        roi_extractor.extract_from_detections(orig_img, detections, image_path, overlay_img=clover_mask)

    # Save normal inference overlay as before
    out_path = os.path.join(config.output_dir, os.path.basename(image_path))
    os.makedirs(config.output_dir, exist_ok=True)
    cv2.imwrite(out_path, combined_with_boxes)
    print(f"[info] Saved: {out_path}")


if __name__ == "__main__":
    main()
