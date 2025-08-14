import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from src.models import create_smp_model, create_effdet_model
from src.transforms import get_val_seg_transforms, get_val_obj_transforms
from effdet import DetBenchPredict
from collections import OrderedDict
import torch.nn.functional as F
from torch import argmax

# ----------------- Main -----------------
def main():
    config = OmegaConf.load("inference_config.yaml")

    color_map = np.array(config.color_map, dtype=np.uint8)
    bbox_color_map = {k: tuple(v) for k, v in config.bbox_color_map.items()}

    # transforms (use the same val transforms as training)
    seg_transforms = get_val_seg_transforms(resize=(1024, 1024))
    obj_transforms = get_val_obj_transforms(resize=(512, 512))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seg_model, det_model = load_models(config, device)

    for img_path in config.image_paths:
        process_image(img_path, seg_model, det_model, seg_transforms, obj_transforms, color_map, bbox_color_map, config, device)


# ----------------- Drawing -----------------
def draw_segmentation_overlay(image, preds, color_map, alpha=0.5, beta=0.5, gamma=0.1):
    # preds: (H, W), uint8
    color_mask = color_map[preds]  # (H, W, 3)
    color_mask = color_mask.astype(np.uint8)

    # Ensure same size as original image
    if color_mask.shape[:2] != image.shape[:2]:
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(image.copy(), alpha, color_mask, beta, gamma)
    return overlay


def draw_bounding_boxes(image, bboxes, bbox_color_map, scale_factor=0.005, padding_factor=0.005, font_thickness_factor=0.001):
    """
    bboxes: list of tuples (label_str, x1, y1, x2, y2, score)
    """
    h = image.shape[0]
    for item in bboxes:
        # allow both (label,x1,y1,x2,y2,score) and (label,x1,y1,x2,y2)
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

    # Segmentation model
    if getattr(config, "enable_segmentation", False):
        seg_model = create_smp_model(config=config.segmentation.model_config)
        checkpoint = torch.load(config.segmentation.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        seg_model.load_state_dict(state)
        seg_model.to(device)
        seg_model.eval()
        print("[info] loaded segmentation model")

    # Detection model
    if getattr(config, "enable_detection", False):
        # Detection config
        # Create training model (no bench_type)
        # no need to do this
        det_cfg = {
            "image_size": tuple(config.detection.get("input_size", (512, 512))),
            "architecture": config.detection.get("architecture", "tf_efficientdet_d0"),
            "pretrained": config.detection.get("pretrained", False),
            "num_classes": int(config.detection.get("num_classes", 3)),
            "max_det_per_image": int(config.detection.get("max_det_per_image", 20)),
        }

        train_model = create_effdet_model(**det_cfg)  # returns DetBenchTrain

        # Load checkpoint
        checkpoint = torch.load(config.detection.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Normalize keys
        new_state = OrderedDict()
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            new_state[nk] = v

        train_model.load_state_dict(new_state, strict=False)

        # Wrap for inference
        det_model = DetBenchPredict(train_model)  # now forward() only needs input
        det_model.eval()
        det_model.to(device)
        print("[info] loaded detection model (DetBenchPredict)")


    return seg_model, det_model


# ----------------- Segmentation inference (training-like + accurate upsampling) -----------------
def run_segmentation_inference(seg_model, img_tensor, device, original_size=None):
    """
    img_tensor: torch tensor (C,H,W) already produced by your val transforms (no batch dim)
    original_size: (H_orig, W_orig) to upsample logits before argmax (recommended)
    """
    seg_model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)  # (1,C,H,W)
        logits = seg_model(x)  # (1, num_classes, H_model, W_model)

        preds = argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # if original_size is not None:
            # upsample logits to original resolution before argmax (more accurate apperently)
            # logits = F.interpolate(logits, size=(original_size[0], original_size[1]), mode="bilinear", align_corners=False)

    return preds


# ----------------- Detection inference -----------------
def run_detection_inference(model, image_bgr, device, score_thresh=0.3):
    model.eval()
    with torch.no_grad():
        h0, w0 = image_bgr.shape[:2]

        # Resize to model input
        img_size = 512
        try:
            cfg = getattr(model, "model", None)
            if cfg is not None and hasattr(cfg, "config") and hasattr(cfg.config, "image_size"):
                img_size = int(max(cfg.config.image_size)) if isinstance(cfg.config.image_size, (list, tuple)) else int(cfg.config.image_size)
        except Exception:
            img_size = 512

        image_rgb = image_bgr[:, :, ::-1]
        resized = cv2.resize(image_rgb, (img_size, img_size))
        inp = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        inp = (inp - mean) / std
        inp = inp.unsqueeze(0).to(device)

        # Forward pass
        out = model(inp)

        # Parse output
        if isinstance(out, dict) and all(k in out for k in ("boxes", "scores", "labels")):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy().astype(int)
        else:
            # fallback for legacy outputs
            arr = out.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.size == 0:
                return []
            boxes = arr[:, :4]
            scores = arr[:, 4]
            labels = arr[:, 5].astype(int)

        # Filter by score
        keep = scores >= score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Rescale boxes to original image
        sx = w0 / img_size
        sy = h0 / img_size
        if boxes.size > 0:
            boxes[:, [0,2]] = np.clip(np.round(boxes[:, [0,2]] * sx), 0, w0-1)
            boxes[:, [1,3]] = np.clip(np.round(boxes[:, [1,3]] * sy), 0, h0-1)
            boxes = boxes.astype(int)

        # Map labels
        class_names = getattr(config.detection, "class_names", None)
        results = []
        for (x1, y1, x2, y2), cid, sc in zip(boxes, labels, scores):
            label_str = str(cid)
            if class_names is not None and 0 <= cid < len(class_names):
                label_str = class_names[cid]
            results.append((label_str, int(x1), int(y1), int(x2), int(y2), float(sc)))

        return results


# ----------------- Process image -----------------
def process_image(image_path, seg_model, det_model, seg_transforms, obj_transforms, color_map, bbox_color_map, config, device):
    orig_img = cv2.imread(image_path ,cv2.IMREAD_COLOR_RGB)
    if orig_img is None:
        print(f"[warn] can't read {image_path}")
        return
    orig_h, orig_w = orig_img.shape[:2]

    combined = orig_img.copy()

    # Segmentation: use val transforms to get tensor, then run inference and upsample to original size
    if seg_model is not None:
        sample = seg_transforms(image=orig_img, mask=None)
        img_tensor = sample["image"]  # (C,H_model,W_model) as torch tensor
        seg_mask = run_segmentation_inference(seg_model, img_tensor, device, original_size=(orig_h, orig_w))
        combined = draw_segmentation_overlay(combined, seg_mask, color_map)


    # Detection: run on original image (run_detection_inference resizes internally and rescales back)
    if det_model is not None:
        detections = run_detection_inference(det_model, orig_img, device, score_thresh=getattr(config.detection, "score_thresh", 0.3))
        # draw detections (each: label, x1, y1, x2, y2, score)
        combined = draw_bounding_boxes(combined, detections, bbox_color_map)

    out_path = os.path.join(config.output_dir, os.path.basename(image_path))
    os.makedirs(config.output_dir, exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
