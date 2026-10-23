import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(CURRENT_DIR, "models", "weapon_detection_yolov8.onnx")
MODEL2_PATH = os.path.join(CURRENT_DIR, "models", "weapon_detection_threat.onnx")


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resizes and pads image while maintaining aspect ratio (YOLO standard letterbox)."""
    shape = im.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class WeaponDetector:
    def __init__(self, conf_threshold: float = 0.28, nms_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net1 = None
        self.net2 = None

        # Load Model 1: YOLOv8 Pistol & Knife (2 classes)
        if os.path.exists(MODEL1_PATH):
            try:
                self.net1 = cv2.dnn.readNetFromONNX(MODEL1_PATH)
                print("[WeaponDetector] Model 1 (YOLOv8 Pistols & Knives) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 1: {e}")

        # Load Model 2: YOLOv8 High-Precision Threat Detector (Guns, Knives, Grenades)
        if os.path.exists(MODEL2_PATH):
            try:
                self.net2 = cv2.dnn.readNetFromONNX(MODEL2_PATH)
                print("[WeaponDetector] Model 2 (Threat Ensemble: Guns, Knives, Grenades) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 2: {e}")

    def set_threshold(self, threshold: float):
        self.conf_threshold = max(0.15, min(0.90, threshold))

    def detect_weapons(self, frame_bgr: np.ndarray, custom_conf: float = None) -> List[Dict[str, Any]]:
        """
        Detects weapons in the given frame using letterbox preprocessing and multi-model ensemble.
        Returns:
        [
            {
                "class": "pistol" | "gun" | "knife" | "grenade",
                "label": "Pistol / Firearm" | "Knife / Blade",
                "confidence": 88.5,
                "bbox": [x, y, w, h]
            }
        ]
        """
        if frame_bgr is None or (self.net1 is None and self.net2 is None):
            return []

        h, w = frame_bgr.shape[:2]
        conf_thresh = custom_conf if custom_conf is not None else self.conf_threshold

        # Letterbox image to 640x640 maintaining aspect ratio
        lb_img, r, (dw, dh) = letterbox(frame_bgr, (640, 640))
        blob = cv2.dnn.blobFromImage(lb_img, 1 / 255.0, (640, 640), swapRB=True, crop=False)

        candidate_boxes = []
        candidate_confs = []
        candidate_classes = []
        candidate_labels = []

        # Model 1 Inference (0: pistol, 1: knife)
        if self.net1 is not None:
            self.net1.setInput(blob)
            preds1 = self.net1.forward()[0].T
            for row in preds1:
                scores = row[4:]  # pistol (0), knife (1)
                best_cls = int(np.argmax(scores))
                score = float(scores[best_cls])
                if score >= conf_thresh:
                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    orig_cx = (cx - dw) / r
                    orig_cy = (cy - dh) / r
                    orig_w = bw / r
                    orig_h = bh / r
                    left = int(max(0, orig_cx - orig_w / 2.0))
                    top = int(max(0, orig_cy - orig_h / 2.0))
                    width = int(min(w - left, orig_w))
                    height = int(min(h - top, orig_h))

                    cls_name = "pistol" if best_cls == 0 else "knife"
                    label = "Pistol / Firearm" if best_cls == 0 else "Knife / Blade"

                    candidate_boxes.append([left, top, width, height])
                    candidate_confs.append(score)
                    candidate_classes.append(cls_name)
                    candidate_labels.append(label)

        # Model 2 Inference (0: Gun, 1: explosion, 2: grenade, 3: knife)
        if self.net2 is not None:
            self.net2.setInput(blob)
            preds2 = self.net2.forward()[0].T
            for row in preds2:
                # We care about 0 (Gun), 2 (grenade), 3 (knife)
                scores = [float(row[4]), 0.0, float(row[6]), float(row[7])]
                best_cls = int(np.argmax(scores))
                score = float(scores[best_cls])
                if score >= conf_thresh:
                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    orig_cx = (cx - dw) / r
                    orig_cy = (cy - dh) / r
                    orig_w = bw / r
                    orig_h = bh / r
                    left = int(max(0, orig_cx - orig_w / 2.0))
                    top = int(max(0, orig_cy - orig_h / 2.0))
                    width = int(min(w - left, orig_w))
                    height = int(min(h - top, orig_h))

                    if best_cls == 0:
                        cls_name = "pistol"
                        label = "Pistol / Firearm"
                    elif best_cls == 3:
                        cls_name = "knife"
                        label = "Knife / Blade"
                    elif best_cls == 2:
                        cls_name = "grenade"
                        label = "Explosive / Grenade"
                    else:
                        continue

                    candidate_boxes.append([left, top, width, height])
                    candidate_confs.append(score)
                    candidate_classes.append(cls_name)
                    candidate_labels.append(label)

        if not candidate_boxes:
            return []

        # Merge candidate detections with Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_confs, conf_thresh, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    "class": candidate_classes[idx],
                    "label": candidate_labels[idx],
                    "confidence": round(float(candidate_confs[idx]) * 100, 1),
                    "bbox": candidate_boxes[idx]
                })

        return results


weapon_detector = WeaponDetector(conf_threshold=0.28)
