import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from src.models import create_smp_model, create_fasterrcnn_model
from src.transforms import get_val_seg_transforms, get_val_obj_transforms
from collections import OrderedDict
from torch import argmax

# ----------------- Main -----------------
def main():
    config = OmegaConf.load("inference_config.yaml")

    color_map = np.array(config.color_map, dtype=np.uint8)
    bbox_color_map = {k: tuple(v) for k, v in config.bbox_color_map.items()}

    # Transforms (use same val transforms as training)
    seg_transforms = get_val_seg_transforms(resize=(1024, 1024))
    det_cfg_input_size = tuple(config.detection.get("input_size", (1024, 1024)))
    obj_transforms = get_val_obj_transforms(resize=det_cfg_input_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seg_model, det_model = load_models(config, device)

    for img_path in config.image_paths:
        process_image(
            img_path,
            seg_model,
            det_model,
            seg_transforms,
            obj_transforms,
            color_map,
            bbox_color_map,
            config,
            device
        )


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


# ----------------- Model loading -----------------
def load_models(config, device):
    seg_model, det_model = None, None

    if getattr(config, "enable_segmentation", False):
        seg_model = create_smp_model(config=config.segmentation.model_config)
        checkpoint = torch.load(config.segmentation.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        seg_model.load_state_dict(state)
        seg_model.to(device).eval()
        print("[info] loaded segmentation model")

    if getattr(config, "enable_detection", False):
        det_cfg = {
            "architecture": config.detection.get("architecture", "fasterrcnn_resnet50_fpn"),
            "pretrained": config.detection.get("pretrained", True),
            "num_classes": int(config.detection.get("num_classes", 2)),  # includes background
            "max_det_per_image": int(config.detection.get("max_det_per_image", 20)),
            "image_size": tuple(config.detection.get("input_size", (1024, 1024)))
        }

        det_model = create_fasterrcnn_model(**det_cfg)
        checkpoint = torch.load(config.detection.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # strip prefixes if trained with DataParallel
        new_state = OrderedDict()
        for k, v in state_dict.items():
            nk = k.replace("module.", "").replace("model.", "")
            new_state[nk] = v

        det_model.load_state_dict(new_state, strict=False)
        det_model.to(device).eval()
        print("[info] loaded detection model (Faster R-CNN)")

    return seg_model, det_model


# ----------------- Segmentation inference -----------------
def run_segmentation_inference(seg_model, img_tensor, device, original_size=None):
    seg_model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        logits = seg_model(x)
        preds = argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return preds


# ----------------- Detection inference -----------------
def run_detection_inference(model, orig_img, obj_transforms, device, score_thresh=0.0, label_map=None):
    """
    Run Faster R-CNN inference on a single image and rescale boxes back to original size.
    """
    model.eval()

    orig_h, orig_w = orig_img.shape[:2]

    # Apply transforms (resizing happens here)
    sample = obj_transforms(image=orig_img, bboxes=[], labels=[])
    image_tensor = sample["image"].to(device)

    # Get resized image size after transform
    resized_h, resized_w = image_tensor.shape[1:]  # C, H, W → take H, W

    with torch.no_grad():
        outputs = model([image_tensor])
        outputs = outputs[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        # Compute scaling factors
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        # Rescale boxes back to original image size
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        boxes = boxes.astype(int)

        if label_map is None:
            label_map = {i: f"class_{i}" for i in range(1, model.num_classes)}

        results = []
        for (x1, y1, x2, y2), cid, sc in zip(boxes, labels, scores):
            if sc >= score_thresh:  # apply threshold here
                label_str = label_map.get(cid, f"class_{cid}")
                results.append((label_str, int(x1), int(y1), int(x2), int(y2), float(sc)))

    return results


# ----------------- Process image -----------------
def process_image(image_path, seg_model, det_model, seg_transforms, obj_transforms, color_map, bbox_color_map, config, device):
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    if orig_img is None:
        print(f"[warn] can't read {image_path}")
        return

    combined = orig_img.copy()

    if seg_model is not None:
        sample = seg_transforms(image=orig_img, mask=None)
        img_tensor = sample["image"]
        seg_mask = run_segmentation_inference(seg_model, img_tensor, device)
        combined = draw_segmentation_overlay(combined, seg_mask, color_map)

    if det_model is not None:
        detections = run_detection_inference(
            det_model,
            orig_img,
            obj_transforms=obj_transforms,
            device=device,
            score_thresh=getattr(config.detection, "score_thresh", 0.0),
            label_map={1: "quadrat_corner"}  # update with your class labels
        )

        if len(detections) == 0:
            print(f"[info] No detections found for {image_path}")
        else:
            print(f"[info] Detections for {image_path}:")
            for det in detections:
                label, x1, y1, x2, y2, score = det
                print(f"   {label} | box=({x1},{y1},{x2},{y2}) | score={score:.3f}")

        combined = draw_bounding_boxes(combined, detections, bbox_color_map)

    out_path = os.path.join(config.output_dir, os.path.basename(image_path))
    os.makedirs(config.output_dir, exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()