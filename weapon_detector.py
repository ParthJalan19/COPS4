import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEAPON_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "weapon_detection_yolov8.onnx")

WEAPON_CLASSES = ["pistol", "knife"]

class WeaponDetector:
    def __init__(self, conf_threshold: float = 0.40, nms_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None

        if os.path.exists(WEAPON_MODEL_PATH):
            try:
                self.net = cv2.dnn.readNetFromONNX(WEAPON_MODEL_PATH)
                print("[WeaponDetector] YOLOv8 Weapon Detection Model loaded successfully.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading model: {e}")
        else:
            print(f"[WeaponDetector] Warning: Weapon model not found at {WEAPON_MODEL_PATH}")

    def detect_weapons(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects weapons (pistol, knife) in the given camera frame.
        Returns list of detected weapons:
        [
            {
                "class": "pistol" | "knife",
                "label": "Pistol / Gun" | "Knife / Blade",
                "confidence": 88.5,
                "bbox": [x, y, w, h]
            }
        ]
        """
        if self.net is None or frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]

        # YOLOv8 expects 640x640 RGB image normalized to [0, 1]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Output shape is (1, 6, 8400). Transpose to (8400, 6)
        preds = outputs[0].T

        boxes = []
        confidences = []
        class_ids = []

        x_scale = w / 640.0
        y_scale = h / 640.0

        for row in preds:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if score >= self.conf_threshold:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]

                left = int((cx - bw / 2.0) * x_scale)
                top = int((cy - bh / 2.0) * y_scale)
                width = int(bw * x_scale)
                height = int(bh * y_scale)

                # Clamp to frame boundary
                left = max(0, left)
                top = max(0, top)
                width = min(w - left, width)
                height = min(h - top, height)

                boxes.append([left, top, width, height])
                confidences.append(score)
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression to eliminate overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid = class_ids[idx]
                raw_cls = WEAPON_CLASSES[cid] if cid < len(WEAPON_CLASSES) else "weapon"
                friendly_label = "Pistol / Firearm" if raw_cls == "pistol" else "Knife / Blade"

                results.append({
                    "class": raw_cls,
                    "label": friendly_label,
                    "confidence": round(float(confidences[idx]) * 100, 1),
                    "bbox": boxes[idx]
                })

        return results


weapon_detector = WeaponDetector()
