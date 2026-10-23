import os
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "person_detection_yolov8.onnx")


class BodyDetector:
    """
    High-accuracy, lightweight YOLOv8n Full-Body Human Detector.
    Runs on CPU with letterbox preprocessing and vectorized candidate filtering.
    Detects full human bodies (head to toe), calculates posture (standing/seated),
    and associates with face & weapon detections.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Person detection model not found at {model_path}")

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_size = (640, 640)
        print(f"[BodyDetector] Loaded YOLOv8 Full Body Detector from {os.path.basename(model_path)}")

    def _letterbox(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Pad image to 640x640 maintaining aspect ratio with uniform gray padding (114).
        """
        h0, w0 = img_bgr.shape[:2]
        r = min(self.input_size[1] / h0, self.input_size[0] / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pad_w, pad_h = (self.input_size[0] - nw) // 2, (self.input_size[1] - nh) // 2

        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
        return canvas, r, pad_w, pad_h

    def detect_bodies(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.38,
        nms_thresh: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Detects all full human bodies in the given BGR frame.
        Returns a list of detected bodies with bounding boxes, confidence, and posture.
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h0, w0 = img_bgr.shape[:2]
        canvas, r, pad_w, pad_h = self._letterbox(img_bgr)

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        preds = self.net.forward()  # Shape: (1, 84, 8400)

        # Class 0 in COCO is 'person'
        person_scores = preds[0, 4, :]  # Shape: (8400,)
        valid_idx = np.where(person_scores >= conf_thresh)[0]
        if len(valid_idx) == 0:
            return []

        cands = preds[0, :, valid_idx]  # Shape: (N, 84)
        scores = person_scores[valid_idx]

        # Rescale from 640x640 letterbox coordinates back to original frame
        cx = (cands[:, 0] - pad_w) / r
        cy = (cands[:, 1] - pad_h) / r
        w = cands[:, 2] / r
        h = cands[:, 3] / r

        x1 = np.clip(cx - w / 2.0, 0, w0)
        y1 = np.clip(cy - h / 2.0, 0, h0)
        w = np.clip(w, 0, w0 - x1)
        h = np.clip(h, 0, h0 - y1)

        boxes = [
            [int(x1[i]), int(y1[i]), int(w[i]), int(h[i])]
            for i in range(len(valid_idx))
        ]

        # Filter out extremely tiny noise (e.g. smaller than 20x40)
        valid_boxes = []
        valid_conf = []
        for b, s in zip(boxes, scores):
            if b[2] >= 20 and b[3] >= 35:
                valid_boxes.append(b)
                valid_conf.append(float(s))

        if not valid_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(valid_boxes, valid_conf, conf_thresh, nms_thresh)
        results = []

        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, bw, bh = valid_boxes[idx]
                score = valid_conf[idx]
                aspect_ratio = bh / max(1, bw)
                posture = "Standing" if aspect_ratio >= 1.35 else "Seated / Bent"

                results.append({
                    "bbox": [bx, by, bw, bh],
                    "confidence": int(round(score * 100)),
                    "class": "person",
                    "label": "Person",
                    "posture": posture,
                    "upper_body": [bx, by, bw, int(bh * 0.45)],
                    "lower_body": [bx, by + int(bh * 0.45), bw, int(bh * 0.55)]
                })

        return results
