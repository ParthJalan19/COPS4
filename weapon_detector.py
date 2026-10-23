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
    def __init__(self, conf_threshold: float = 0.20, nms_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net1 = None
        self.net2 = None

        # Model 1: Specialized YOLOv8 Pistol & Knife Detector
        if os.path.exists(MODEL1_PATH):
            try:
                self.net1 = cv2.dnn.readNetFromONNX(MODEL1_PATH)
                print("[WeaponDetector] Model 1 (Specialized YOLOv8 Pistols & Knives) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 1: {e}")

        # Model 2: Real-World Threat Detector (Guns, Knives, Grenades)
        if os.path.exists(MODEL2_PATH):
            try:
                self.net2 = cv2.dnn.readNetFromONNX(MODEL2_PATH)
                print("[WeaponDetector] Model 2 (Threat Ensemble: Guns, Knives, Grenades) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 2: {e}")

    def set_threshold(self, threshold: float):
        self.conf_threshold = max(0.12, min(0.90, threshold))

    def _infer_image(self, img_bgr: np.ndarray, conf_thresh: float) -> Tuple[List, List, List, List]:
        """Runs vectorized inference on a single image and returns candidate detections."""
        h, w = img_bgr.shape[:2]
        lb_img, r, (dw, dh) = letterbox(img_bgr, (640, 640))
        blob = cv2.dnn.blobFromImage(lb_img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)

        boxes = []
        confs = []
        classes = []
        labels = []

        # Model 1 (0: pistol, 1: knife)
        if self.net1 is not None:
            self.net1.setInput(blob)
            p1 = self.net1.forward()[0].T
            s1 = p1[:, 4:6]
            c1 = np.argmax(s1, axis=1)
            v1 = s1[np.arange(len(s1)), c1]
            m1 = v1 >= conf_thresh

            if np.any(m1):
                valid_p = p1[m1]
                valid_c = c1[m1]
                valid_s = v1[m1]

                cx = (valid_p[:, 0] - dw) / r
                cy = (valid_p[:, 1] - dh) / r
                bw = valid_p[:, 2] / r
                bh = valid_p[:, 3] / r

                left = np.clip(cx - bw / 2.0, 0, w).astype(int)
                top = np.clip(cy - bh / 2.0, 0, h).astype(int)
                width = np.clip(bw, 1, w - left).astype(int)
                height = np.clip(bh, 1, h - top).astype(int)

                for i in range(len(valid_s)):
                    boxes.append([int(left[i]), int(top[i]), int(width[i]), int(height[i])])
                    confs.append(float(valid_s[i]))
                    cls_name = "pistol" if valid_c[i] == 0 else "knife"
                    classes.append(cls_name)
                    labels.append("Pistol / Firearm" if cls_name == "pistol" else "Knife / Blade")

        # Model 2 (0: Gun, 2: grenade, 3: knife)
        if self.net2 is not None:
            self.net2.setInput(blob)
            p2 = self.net2.forward()[0].T
            s2 = np.stack([p2[:, 4], np.zeros(len(p2)), p2[:, 6], p2[:, 7]], axis=1)
            c2 = np.argmax(s2, axis=1)
            v2 = s2[np.arange(len(s2)), c2]
            m2 = v2 >= conf_thresh

            if np.any(m2):
                valid_p = p2[m2]
                valid_c = c2[m2]
                valid_s = v2[m2]

                cx = (valid_p[:, 0] - dw) / r
                cy = (valid_p[:, 1] - dh) / r
                bw = valid_p[:, 2] / r
                bh = valid_p[:, 3] / r

                left = np.clip(cx - bw / 2.0, 0, w).astype(int)
                top = np.clip(cy - bh / 2.0, 0, h).astype(int)
                width = np.clip(bw, 1, w - left).astype(int)
                height = np.clip(bh, 1, h - top).astype(int)

                for i in range(len(valid_s)):
                    cid = valid_c[i]
                    if cid == 0:
                        cls_name, lbl = "pistol", "Pistol / Firearm"
                    elif cid == 3:
                        cls_name, lbl = "knife", "Knife / Blade"
                    elif cid == 2:
                        cls_name, lbl = "grenade", "Explosive / Grenade"
                    else:
                        continue

                    boxes.append([int(left[i]), int(top[i]), int(width[i]), int(height[i])])
                    confs.append(float(valid_s[i]))
                    classes.append(cls_name)
                    labels.append(lbl)

        return boxes, confs, classes, labels

    def detect_weapons(self, frame_bgr: np.ndarray, custom_conf: float = None) -> List[Dict[str, Any]]:
        """
        Detects weapons using vectorized inference and multi-scale ROI refinement.
        """
        if frame_bgr is None or (self.net1 is None and self.net2 is None):
            return []

        h, w = frame_bgr.shape[:2]
        conf_thresh = custom_conf if custom_conf is not None else self.conf_threshold

        # Pass 1: Full-Frame Inference
        all_boxes, all_confs, all_classes, all_labels = self._infer_image(frame_bgr, conf_thresh)

        # Pass 2: Multi-Scale Hand/Held ROI Scan
        # In webcam/CCTV feeds, handguns and knives held by a person are often smaller objects.
        # If no weapon detected on full frame, scan central 75% region with 1.33x magnification
        if len(all_boxes) == 0 and h >= 240 and w >= 320:
            roi_y1 = int(h * 0.15)
            roi_y2 = int(h * 0.95)
            roi_x1 = int(w * 0.12)
            roi_x2 = int(w * 0.88)
            crop = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

            if crop.size > 0:
                c_boxes, c_confs, c_classes, c_labels = self._infer_image(crop, conf_thresh)
                for cb, sc, cl, lb in zip(c_boxes, c_confs, c_classes, c_labels):
                    # Translate crop box back to full frame
                    all_boxes.append([cb[0] + roi_x1, cb[1] + roi_y1, cb[2], cb[3]])
                    all_confs.append(sc)
                    all_classes.append(cl)
                    all_labels.append(lb)

        if not all_boxes:
            return []

        # Non-Maximum Suppression (NMS) to eliminate overlaps
        indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, conf_thresh, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    "class": all_classes[idx],
                    "label": all_labels[idx],
                    "confidence": round(float(all_confs[idx]) * 100, 1),
                    "bbox": all_boxes[idx]
                })

        return results


weapon_detector = WeaponDetector(conf_threshold=0.20)
