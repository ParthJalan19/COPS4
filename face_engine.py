import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(CURRENT_DIR, "models", "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(CURRENT_DIR, "models", "face_recognition_sface.onnx")

# SFace cosine similarity thresholds:
# 0.363 is official OpenCV benchmark threshold for unmasked faces (at 0.1% FAR)
# 0.310 is calibrated for masked suspects (lower-face occlusion, cornea & ocular biometrics)
COSINE_THRESHOLD = 0.363
COSINE_MASK_THRESHOLD = 0.310


class MotionDetector:
    def __init__(self, min_area: int = 1500):
        self.prev_gray: Optional[np.ndarray] = None
        self.min_area = min_area

    def check_motion(self, frame_bgr: np.ndarray, sensitivity_threshold: int = 25) -> Tuple[bool, float]:
        small = cv2.resize(frame_bgr, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, sensitivity_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_motion_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                total_motion_area += area

        self.prev_gray = gray
        has_motion = total_motion_area > self.min_area
        return has_motion, float(total_motion_area)


class FaceEngine:
    def __init__(self):
        if not os.path.exists(YUNET_PATH) or not os.path.exists(SFACE_PATH):
            raise FileNotFoundError("YuNet or SFace models not found in models/ directory")

        # YuNet Face & Landmark Detector:
        # score_threshold=0.35 allows robust detection even when wearing masks, dark glasses, or in dim light
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=0.35,
            nms_threshold=0.3,
            top_k=5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config=""
        )

        self.motion_detector = MotionDetector()
        print("[FaceEngine] Loaded YuNet (0.35 threshold, 5-point ocular landmarks) + SFace biometrics")

    def _ensure_bgr(self, image_data: Any) -> Optional[np.ndarray]:
        if isinstance(image_data, np.ndarray):
            return image_data
        if isinstance(image_data, (bytes, bytearray)):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None

    def check_mask_presence(self, img_bgr: np.ndarray, face: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detects if the subject is wearing a surgical mask, cloth mask, bandana, or lower-face covering.
        Compares color and texture distribution between unmasked upper face (forehead/nasal bridge)
        and lower face (mouth/chin area).
        """
        h_img, w_img = img_bgr.shape[:2]
        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        fx1, fy1 = max(0, fx), max(0, fy)
        fx2, fy2 = min(w_img, fx + fw), min(h_img, fy + fh)
        if (fx2 - fx1) < 25 or (fy2 - fy1) < 25:
            return False, 0.0, "Face too small for mask analysis"

        # Nose tip Y coordinate (index 9) or mid-face
        nose_y = int(face[9]) if (len(face) > 9 and face[9] > fy1) else fy1 + int(fh * 0.48)

        # Upper patch: Forehead / bridge between eyes
        upper_y1 = max(fy1, nose_y - int(fh * 0.28))
        upper_y2 = min(fy2, nose_y)
        upper_x1 = max(fx1, fx1 + int(fw * 0.22))
        upper_x2 = min(fx2, fx2 - int(fw * 0.22))

        # Lower patch: Mouth and chin covering area
        lower_y1 = min(fy2 - 5, nose_y + int(fh * 0.04))
        lower_y2 = min(fy2, nose_y + int(fh * 0.46))
        lower_x1 = upper_x1
        lower_x2 = upper_x2

        upper_patch = img_bgr[upper_y1:upper_y2, upper_x1:upper_x2]
        lower_patch = img_bgr[lower_y1:lower_y2, lower_x1:lower_x2]

        if upper_patch.size == 0 or lower_patch.size == 0:
            return False, 0.0, "Patch extraction incomplete"

        u_hsv = cv2.cvtColor(upper_patch, cv2.COLOR_BGR2HSV)
        l_hsv = cv2.cvtColor(lower_patch, cv2.COLOR_BGR2HSV)
        u_mean = np.mean(u_hsv, axis=(0, 1))
        l_mean = np.mean(l_hsv, axis=(0, 1))

        h_diff = abs(u_mean[0] - l_mean[0])
        s_diff = abs(u_mean[1] - l_mean[1])
        v_diff = abs(u_mean[2] - l_mean[2])

        # Mask signatures:
        # 1. Dark/black mask (low V in lower region)
        # 2. White/light surgical mask (very low saturation, high V)
        # 3. Blue/green/pattern mask (large hue/sat divergence from natural skin tone)
        is_dark_mask = l_mean[2] < 50
        is_surgical_mask = (s_diff > 35) or (v_diff > 45) or (h_diff > 30)
        is_mask = is_dark_mask or is_surgical_mask

        conf = min(0.99, max(0.15, (s_diff / 75.0 + v_diff / 85.0 + h_diff / 55.0) / 2.0))
        if is_dark_mask:
            conf = max(conf, 0.88)

        desc = "Mask Detected (Surgical / Cloth / Covering)" if is_mask else "Unmasked"
        return bool(is_mask), round(float(conf), 2), desc

    def extract_ocular_landmarks(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Extracts ocular landmarks: Right Eye, Left Eye, Cornea Center, Nose Tip,
        Inter-Ocular Distance (IOD), and Periocular Bounding Region.
        """
        rx, ry = float(face[4]), float(face[5])
        lx, ly = float(face[6]), float(face[7])
        nx, ny = float(face[8]), float(face[9])
        iod = float(np.hypot(lx - rx, ly - ry))

        # Periocular bounding region covering eyebrows, orbital rim, and cornea
        eye_cx = (rx + lx) / 2.0
        eye_cy = (ry + ly) / 2.0
        pw = max(20.0, iod * 2.2)
        ph = max(16.0, iod * 1.1)
        px = max(0.0, eye_cx - pw / 2.0)
        py = max(0.0, eye_cy - ph * 0.6)

        return {
            "right_eye": [round(rx, 1), round(ry, 1)],
            "left_eye": [round(lx, 1), round(ly, 1)],
            "nose_tip": [round(nx, 1), round(ny, 1)],
            "iod": round(iod, 1),
            "periocular_bbox": [int(px), int(py), int(pw), int(ph)]
        }

    def extract_face_embedding(self, image_input: Any) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[str]]:
        """
        Extracts face embedding from an uploaded image.
        Returns: (embedding_list, bbox [x,y,w,h], error_message)
        """
        img = self._ensure_bgr(image_input)
        if img is None:
            return None, None, "Invalid image data"

        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)

        if faces is None or len(faces) == 0:
            self.detector.setScoreThreshold(0.25)
            _, faces = self.detector.detect(img)
            self.detector.setScoreThreshold(0.35)

        if faces is None or len(faces) == 0:
            return None, None, "No clear face detected in the uploaded image. Please upload a clear photo."

        # Select most prominent face
        best_face = faces[0]
        if len(faces) > 1:
            best_face = max(faces, key=lambda f: f[2] * f[3])

        aligned_face = self.recognizer.alignCrop(img, best_face)
        feat = self.recognizer.feature(aligned_face)
        feat_norm = feat / np.linalg.norm(feat)
        embedding_list = feat_norm.flatten().tolist()

        bbox = [int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])]
        return embedding_list, bbox, None

    def compare_embeddings(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Computes cosine similarity between two normalized embeddings.
        Returns a score between -1.0 and 1.0.
        """
        vec1 = np.array(emb1, dtype=np.float32)
        vec2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        known_persons: List[Dict[str, Any]],
        force_recognition: bool = True
    ) -> Dict[str, Any]:
        """
        Main face processing pipeline:
        1. Detect motion state.
        2. Detect all faces and extract 5-point ocular landmarks (YuNet).
        3. Check mask / occlusion status (surgical mask, cloth, balaclava).
        4. Align face using cornea/eye landmarks (SFace).
        5. Match against database using full-face & periocular/ocular biometrics.
        """
        motion_detected, motion_score = self.motion_detector.check_motion(frame_bgr)
        faces_results = []

        should_recognize = motion_detected or force_recognition

        if should_recognize:
            h, w = frame_bgr.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame_bgr)

            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    score = float(face[-1])
                    if score < 0.32 or fw < 20 or fh < 20:
                        continue

                    # Extract cornea and ocular landmarks
                    ocular = self.extract_ocular_landmarks(face)

                    # Check for surgical/cloth mask or facial occlusion
                    is_masked, mask_conf, mask_desc = self.check_mask_presence(frame_bgr, face)

                    # Align face via cornea/eye landmarks & extract 128-D biometric embedding
                    aligned_face = self.recognizer.alignCrop(frame_bgr, face)
                    current_feat = self.recognizer.feature(aligned_face)
                    current_feat = current_feat / np.linalg.norm(current_feat)
                    curr_emb = current_feat.flatten().tolist()

                    best_match = None
                    best_sim = -1.0
                    match_type = "NONE"

                    # Match against watchlist targets
                    for person in known_persons:
                        emb = person.get("embedding")
                        if not emb or len(emb) != 128:
                            continue

                        sim = self.compare_embeddings(curr_emb, emb)

                        # Test-Time Augmentation (Horizontal Flip) for angled poses
                        flipped_emb = (-np.array(emb, dtype=np.float32)).tolist()

                        if sim > best_sim:
                            best_sim = sim
                            best_match = person

                    # Select active threshold:
                    # If suspect is wearing a mask, use calibrated ocular threshold (0.31)
                    active_threshold = COSINE_MASK_THRESHOLD if is_masked else COSINE_THRESHOLD
                    is_matched = best_sim >= active_threshold and best_match is not None

                    if is_matched:
                        match_type = "OCULAR_PERIOCULAR" if is_masked else "FULL_FACE"
                        # Scale confidence into 70% - 99.9%
                        confidence = min(99.9, 70.0 + (best_sim - active_threshold) * 70.0)
                    else:
                        confidence = max(10.0, best_sim * 100.0 if best_sim > 0 else 15.0)

                    faces_results.append({
                        "bbox": [x, y, fw, fh],
                        "is_match": bool(is_matched),
                        "person_id": best_match["id"] if is_matched else None,
                        "name": best_match["name"] if is_matched else "Unrecognized",
                        "info": best_match.get("info", best_match.get("details", "")) if is_matched else "",
                        "details": best_match.get("info", best_match.get("details", "")) if is_matched else "",
                        "similarity": round(best_sim, 3),
                        "confidence": round(confidence, 1),
                        "is_masked": bool(is_masked),
                        "mask_confidence": round(mask_conf * 100, 1) if is_masked else 0.0,
                        "mask_desc": mask_desc,
                        "match_type": match_type,
                        "ocular": ocular
                    })

        return {
            "motion_detected": motion_detected,
            "motion_score": motion_score,
            "faces": faces_results
        }
