"""
COPS-4 Optimized & Calibrated Weapon Detection Engine.
Features:
- Dual-Model YOLOv8 Threat Architecture (Specialized Knives/Blades + Pistols/Firearms)
- Multi-Layered False Positive Filtering:
  1. Negative object disambiguation (suppresses cell phones, remotes, cups, bottles via COCO)
  2. Face/Head overlap suppression (prevents face/hair/glasses false alarms)
  3. Actor spatial proximity gating (ensures weapons are within human reach; rejects empty-room clutter)
  4. Geometric aspect-ratio and edge-sliver validation (eliminates borders, poles, blinds)
  5. Cross-class NMS conflict suppression
  6. Multi-scale ROI zoom refinement for distant objects
  7. Multi-frame temporal persistence tracking (requires 3 consecutive frames to confirm)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from weapon_tracker import weapon_tracker

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(CURRENT_DIR, "models", "weapon_detection_yolov8.onnx")
MODEL2_PATH = os.path.join(CURRENT_DIR, "models", "weapon_detection_threat.onnx")

# Limit CPU thread count for OpenCV DNN to prevent starving FastAPI event loop
cv2.setNumThreads(2)


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


def calculate_box_iou(boxA, boxB) -> float:
    """Calculates IoU between two [x, y, w, h] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_box_overlap_ratio(boxA, boxB) -> float:
    """Calculates intersection area relative to boxA's area (how much of boxA is inside boxB)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(1, boxA[2] * boxA[3])
    return float(inter_area) / float(boxA_area)


class WeaponDetector:
    def __init__(
        self,
        conf_threshold: float = 0.42,
        nms_threshold: float = 0.40,
        use_ensemble: bool = True,
        enable_temporal_tracking: bool = True
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_ensemble = use_ensemble
        self.enable_temporal_tracking = enable_temporal_tracking

        # Calibrated baseline confidence thresholds per weapon class (suppresses weak model noise)
        self.class_thresholds = {
            "pistol": 0.42,
            "knife": 0.38,
            "rifle": 0.42,
            "grenade": 0.45
        }

        self.net1 = None
        self.net2 = None

        # Model 1: Primary Specialized YOLOv8 Pistol & Knife Detector
        if os.path.exists(MODEL1_PATH):
            try:
                self.net1 = cv2.dnn.readNetFromONNX(MODEL1_PATH)
                self.net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[WeaponDetector] Model 1 (Specialized YOLOv8 Pistols & Knives) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 1: {e}")

        # Model 2: Threat Ensemble Firearm Secondary Verifier
        if os.path.exists(MODEL2_PATH):
            try:
                self.net2 = cv2.dnn.readNetFromONNX(MODEL2_PATH)
                self.net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[WeaponDetector] Model 2 (Threat Ensemble: Firearm Verifier) loaded.")
            except Exception as e:
                print(f"[WeaponDetector] Error loading Model 2: {e}")

    def set_threshold(self, threshold: float):
        self.conf_threshold = max(0.20, min(0.90, threshold))

    def _is_valid_geometry(self, cls_name: str, bbox: List[int], img_shape: Tuple[int, int]) -> bool:
        """
        Validates bounding box geometry and aspect ratio to filter false positives
        such as background vertical edge artifacts, door frames, or border slivers.
        """
        x, y, w, h = bbox
        ih, iw = img_shape[:2]

        # Minimum physical size requirement: at least 22px on each axis and 550 px area
        if w < 22 or h < 22 or (w * h) < 550:
            return False

        touching_edge = (x <= 3 or y <= 3 or (x + w) >= iw - 3 or (y + h) >= ih - 3)
        aspect_ratio = max(w / max(1, h), h / max(1, w))

        if cls_name in ("pistol", "rifle"):
            # Pistols and handguns have realistic aspect ratios <= 2.6.
            # Long thin rectangles (>2.6) touching image borders are door edges, poles or blinds.
            if aspect_ratio > 2.6:
                return False
            if touching_edge and (w < 45 or h < 45):
                return False

        elif cls_name == "knife":
            # Knives/blades can be elongated along the blade axis, but ratio rarely exceeds 6.5
            if aspect_ratio > 6.5:
                return False
            if touching_edge and min(w, h) < 20:
                return False

        return True

    def _infer_image(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float,
        run_model2: bool = True
    ) -> Tuple[List, List, List, List]:
        """Runs vectorized inference on an image and returns candidate weapon detections."""
        h, w = img_bgr.shape[:2]
        lb_img, r, (dw, dh) = letterbox(img_bgr, (640, 640))
        blob = cv2.dnn.blobFromImage(lb_img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)

        # Scale per-class thresholds based on user-chosen sensitivity
        scale = conf_thresh / 0.42
        eff_pistol = max(0.25, min(0.85, self.class_thresholds["pistol"] * scale))
        eff_knife = max(0.22, min(0.85, self.class_thresholds["knife"] * scale))

        boxes = []
        confs = []
        classes = []
        labels = []

        # Model 1: (0: pistol, 1: knife)
        if self.net1 is not None:
            self.net1.setInput(blob)
            p1 = self.net1.forward()[0].T
            s1 = p1[:, 4:6]
            c1 = np.argmax(s1, axis=1)
            v1 = s1[np.arange(len(s1)), c1]

            thresh_arr1 = np.where(c1 == 0, eff_pistol, eff_knife)
            m1 = v1 >= thresh_arr1

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
                    cls_name = "pistol" if valid_c[i] == 0 else "knife"
                    bbox = [int(left[i]), int(top[i]), int(width[i]), int(height[i])]

                    if not self._is_valid_geometry(cls_name, bbox, (h, w)):
                        continue

                    boxes.append(bbox)
                    confs.append(float(valid_s[i]))
                    classes.append(cls_name)
                    labels.append("Pistol / Firearm" if cls_name == "pistol" else "Knife / Blade")

        # Model 2: Secondary Firearm Verification (channel 4 = firearm)
        if self.net2 is not None and run_model2:
            self.net2.setInput(blob)
            p2 = self.net2.forward()[0].T
            s2_firearm = p2[:, 4]  # Class 0 = Firearm
            m2 = s2_firearm >= eff_pistol

            if np.any(m2):
                valid_p = p2[m2]
                valid_s = s2_firearm[m2]

                cx = (valid_p[:, 0] - dw) / r
                cy = (valid_p[:, 1] - dh) / r
                bw = valid_p[:, 2] / r
                bh = valid_p[:, 3] / r

                left = np.clip(cx - bw / 2.0, 0, w).astype(int)
                top = np.clip(cy - bh / 2.0, 0, h).astype(int)
                width = np.clip(bw, 1, w - left).astype(int)
                height = np.clip(bh, 1, h - top).astype(int)

                for i in range(len(valid_s)):
                    bbox = [int(left[i]), int(top[i]), int(width[i]), int(height[i])]
                    if not self._is_valid_geometry("pistol", bbox, (h, w)):
                        continue

                    # Check if overlaps with an existing detection from Model 1
                    matched_existing = False
                    for b_idx, ex_box in enumerate(boxes):
                        if calculate_box_iou(bbox, ex_box) > 0.40:
                            matched_existing = True
                            if classes[b_idx] == "pistol":
                                confs[b_idx] = max(confs[b_idx], float(valid_s[i]))
                            break

                    if not matched_existing and float(valid_s[i]) >= eff_pistol * 1.15:
                        boxes.append(bbox)
                        confs.append(float(valid_s[i]))
                        classes.append("pistol")
                        labels.append("Pistol / Firearm")

        return boxes, confs, classes, labels

    def detect_weapons(
        self,
        frame_bgr: np.ndarray,
        custom_conf: Optional[float] = None,
        camera_id: str = "cam_main",
        has_actor: bool = True,
        bodies: Optional[List[Dict[str, Any]]] = None,
        faces: Optional[List[Dict[str, Any]]] = None,
        confusers: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detects weapons using vectorized inference, multi-scale ROI refinement,
        geometry validation, negative object disambiguation, actor spatial proximity gating,
        cross-class NMS, and multi-frame temporal tracking.
        """
        if frame_bgr is None or (self.net1 is None and self.net2 is None):
            return []

        h, w = frame_bgr.shape[:2]
        conf_thresh = custom_conf if custom_conf is not None else self.conf_threshold

        # Pass 1: Full-Frame Inference
        all_boxes, all_confs, all_classes, all_labels = self._infer_image(
            frame_bgr, conf_thresh, run_model2=self.use_ensemble
        )

        # Pass 2: Adaptive Multi-Scale ROI Scan
        # If no weapon found in full frame and resolution is sufficient, scan central active area
        if len(all_boxes) == 0 and h >= 240 and w >= 320:
            roi_y1 = int(h * 0.12)
            roi_y2 = int(h * 0.92)
            roi_x1 = int(w * 0.10)
            roi_x2 = int(w * 0.90)
            crop = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

            if crop.size > 0:
                c_boxes, c_confs, c_classes, c_labels = self._infer_image(
                    crop, conf_thresh, run_model2=False
                )
                for cb, sc, cl, lb in zip(c_boxes, c_confs, c_classes, c_labels):
                    all_boxes.append([cb[0] + roi_x1, cb[1] + roi_y1, cb[2], cb[3]])
                    all_confs.append(sc)
                    all_classes.append(cl)
                    all_labels.append(lb)

        if not all_boxes:
            if self.enable_temporal_tracking:
                weapon_tracker.update(camera_id, [])
            return []

        # Standard NMS
        indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, conf_thresh * 0.85, self.nms_threshold)

        nms_boxes = []
        nms_confs = []
        nms_classes = []
        nms_labels = []

        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                nms_boxes.append(all_boxes[idx])
                nms_confs.append(all_confs[idx])
                nms_classes.append(all_classes[idx])
                nms_labels.append(all_labels[idx])

        # Cross-Class Conflict Suppression:
        # If two different classes (e.g. knife and pistol) heavily overlap, keep the one with higher confidence
        final_boxes = []
        final_confs = []
        final_classes = []
        final_labels = []
        suppressed_indices = set()

        for i in range(len(nms_boxes)):
            if i in suppressed_indices:
                continue
            for j in range(i + 1, len(nms_boxes)):
                if j in suppressed_indices:
                    continue
                iou = calculate_box_iou(nms_boxes[i], nms_boxes[j])
                if iou > 0.35:
                    if nms_confs[i] >= nms_confs[j]:
                        suppressed_indices.add(j)
                    else:
                        suppressed_indices.add(i)
                        break

            if i not in suppressed_indices:
                final_boxes.append(nms_boxes[i])
                final_confs.append(nms_confs[i])
                final_classes.append(nms_classes[i])
                final_labels.append(nms_labels[i])

        # --- MULTI-LAYERED FALSE POSITIVE FILTERING ---
        validated_boxes = []
        validated_confs = []
        validated_classes = []
        validated_labels = []

        for box, conf, cls_n, lbl in zip(final_boxes, final_confs, final_classes, final_labels):
            bx, by, bw, bh = box

            # Filter 1: Everyday Object Confuser Suppression (cell phones, remotes, cups, etc.)
            if confusers:
                is_confuser = False
                for c in confusers:
                    overlap = calculate_box_overlap_ratio(box, c["bbox"])
                    iou = calculate_box_iou(box, c["bbox"])
                    if (overlap >= 0.35 or iou >= 0.25) and c.get("confidence", 0) >= 22.0:
                        is_confuser = True
                        break
                if is_confuser:
                    continue

            # Filter 2: Face / Head Overlap Suppression (face occlusions, glasses, hair)
            if faces:
                is_face_overlap = False
                for f in faces:
                    overlap = calculate_box_overlap_ratio(box, f["bbox"])
                    if overlap >= 0.45:
                        is_face_overlap = True
                        break
                if is_face_overlap:
                    continue

            # Filter 3: Actor Proximity & Empty Room Gating
            if not has_actor:
                # In an empty room with no detected human body or face, require high confidence
                if conf < 0.70:
                    continue
            elif bodies and len(bodies) > 0:
                # If humans are present, weapon must be within reachable distance of a human body
                near_human = False
                for b in bodies:
                    px, py, pw, ph = b["bbox"]
                    reach_pad_x = int(pw * 0.45)
                    reach_pad_y = int(ph * 0.25)
                    rx1 = max(0, px - reach_pad_x)
                    ry1 = max(0, py - reach_pad_y)
                    rx2 = px + pw + reach_pad_x
                    ry2 = py + ph + reach_pad_y
                    wcx = bx + bw / 2
                    wcy = by + bh / 2
                    if rx1 <= wcx <= rx2 and ry1 <= wcy <= ry2:
                        near_human = True
                        break
                if not near_human and conf < 0.75:
                    # Weapon candidate is far away on an empty background -> static clutter false alarm
                    continue

            validated_boxes.append(box)
            validated_confs.append(conf)
            validated_classes.append(cls_n)
            validated_labels.append(lbl)

        raw_results = []
        for i in range(len(validated_boxes)):
            raw_results.append({
                "class": validated_classes[i],
                "label": validated_labels[i],
                "confidence": round(float(validated_confs[i]) * 100, 1),
                "bbox": validated_boxes[i]
            })

        # Apply Temporal Association & Tracking
        if self.enable_temporal_tracking:
            tracked_results = weapon_tracker.update(camera_id, raw_results)
            return tracked_results

        return raw_results


weapon_detector = WeaponDetector(conf_threshold=0.42, use_ensemble=True)
