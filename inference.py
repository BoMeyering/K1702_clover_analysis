import os
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from src.models import create_smp_model, create_effdet_model
from src.transforms import get_val_seg_transforms, get_val_obj_transforms
from collections import OrderedDict
from torch import argmax

# ----------------- Main -----------------
def main():
    config = OmegaConf.load("inference_config.yaml")

    color_map = np.array(config.color_map, dtype=np.uint8)
    bbox_color_map = {k: tuple(v) for k, v in config.bbox_color_map.items()}

    # transforms (use the same val transforms as training)
    seg_transforms = get_val_seg_transforms(resize=(1024, 1024))

    # Get detection input size from config
    det_cfg_input_size = tuple(config.detection.get("input_size", (1024, 1024)))
    obj_transforms = get_val_obj_transforms(resize=det_cfg_input_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    seg_model, det_model = load_models(config, device)

    for img_path in config.image_paths:
        process_image(img_path, seg_model, det_model, seg_transforms, obj_transforms, color_map, bbox_color_map, config, device)


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
        seg_model.to(device)
        seg_model.eval()
        print("[info] loaded segmentation model")

    if getattr(config, "enable_detection", False):
        det_cfg = {
            "image_size": tuple(config.detection.get("input_size", (1024, 1024))),
            "architecture": config.detection.get("architecture", "efficientdet_d0"),
            "pretrained": config.detection.get("pretrained", True),
            "num_classes": int(config.detection.get("num_classes", 3)),
            "max_det_per_image": int(config.detection.get("max_det_per_image", 20)),
            "mode": "predict"
        }

        det_model = create_effdet_model(**det_cfg)

        checkpoint = torch.load(config.detection.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        new_state = OrderedDict()
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            new_state[nk] = v

        det_model.load_state_dict(new_state, strict=False)
        det_model.eval()
        det_model.to(device)
        print("[info] loaded detection model (DetBenchPredict via create_effdet_model)")

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
def run_detection_inference(model, image_bgr, device, config=None, score_thresh=0.5):
    model.eval()
    with torch.no_grad():
        h0, w0 = image_bgr.shape[:2]

        img_size_cfg = getattr(model.model.config, "image_size", 1024)
        if hasattr(img_size_cfg, "__len__"):
            img_size = (int(img_size_cfg[1]), int(img_size_cfg[0]))
        else:
            img_size = (int(img_size_cfg), int(img_size_cfg))

        image_resized = cv2.resize(image_bgr, img_size, interpolation=cv2.INTER_LINEAR)
        image_rgb = image_resized[:, :, ::-1].astype(np.float32) / 255.0

        inp = torch.from_numpy(image_rgb).permute(2,0,1).unsqueeze(0)
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        inp = ((inp - mean)/std).to(device)

        out = model(inp)
        out_np = out.cpu().numpy()[0]

        boxes_all = out_np[:, :4]
        scores    = out_np[:, 4]
        labels    = out_np[:, 5].astype(int)

        # Apply score threshold
        keep = scores >= score_thresh
        boxes  = boxes_all[keep, :]
        scores = scores[keep]
        labels = labels[keep]

        # Rescale to original image size
        sx, sy = w0 / img_size[0], h0 / img_size[1]
        if boxes.size > 0:
            boxes[:, [0,2]] = np.clip(np.round(boxes[:, [0,2]] * sx), 0, w0-1)
            boxes[:, [1,3]] = np.clip(np.round(boxes[:, [1,3]] * sy), 0, h0-1)
            boxes = boxes.astype(int)

        # Hardcoded label mapping
        label_map = {
            1: "clover",
            2: "quadrat",
            3: "quadrat_corner"
        }

        results = []
        # the model expects y1,x1,y2,x2 so reorder and feed it to the model
        for (x1,y1,x2,y2), cid, sc in zip(boxes, labels, scores):
            label_str = label_map.get(cid, f"class_{cid}")
            results.append((label_str, int(x1), int(y1), int(x2), int(y2), float(sc)))

        return results


# ----------------- Process image -----------------
def process_image(image_path, seg_model, det_model, seg_transforms, obj_transforms, color_map, bbox_color_map, config, device):
    orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        print(f"[warn] can't read {image_path}")
        return
    orig_h, orig_w = orig_img.shape[:2]

    combined = orig_img.copy()

    if seg_model is not None:
        sample = seg_transforms(image=orig_img, mask=None)
        img_tensor = sample["image"]
        seg_mask = run_segmentation_inference(seg_model, img_tensor, device, original_size=(orig_h, orig_w))
        combined = draw_segmentation_overlay(combined, seg_mask, color_map)

    if det_model is not None:
        detections = run_detection_inference(
            det_model,
            orig_img,  # <<< pass original
            device,
            score_thresh=getattr(config.detection, "score_thresh", 0.3)
        )

        # --- Print raw detections ---
        if len(detections) == 0:
            print(f"[info] No detections found for {image_path}")
        else:
            print(f"[info] Detections for {image_path}:")
            for det in detections:
                label, x1, y1, x2, y2, score = det
                print(f"   {label} | box=({x1},{y1},{x2},{y2}) | score={score:.3f}")

        # --- Draw detections ---
        combined = draw_bounding_boxes(combined, detections, bbox_color_map)

    out_path = os.path.join(config.output_dir, os.path.basename(image_path))
    os.makedirs(config.output_dir, exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
