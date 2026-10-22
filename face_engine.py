import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(CURRENT_DIR, "models", "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(CURRENT_DIR, "models", "face_recognition_sface.onnx")

# SFace cosine similarity threshold (0.363 is official threshold, 0.40 provides high precision)
COSINE_THRESHOLD = 0.40


class MotionDetector:
    def __init__(self, min_area: int = 1500):
        self.prev_gray: Optional[np.ndarray] = None
        self.min_area = min_area

    def check_motion(self, frame_bgr: np.ndarray, sensitivity_threshold: int = 25) -> Tuple[bool, float]:
        # Resize for fast motion processing
        small = cv2.resize(frame_bgr, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        # Compute absolute difference
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

        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=0.55,
            nms_threshold=0.3,
            top_k=5000
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config=""
        )

        self.motion_detector = MotionDetector()

    def _ensure_bgr(self, image_data: Any) -> Optional[np.ndarray]:
        if isinstance(image_data, np.ndarray):
            return image_data
        if isinstance(image_data, (bytes, bytearray)):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None

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
            # Try lower threshold if no face found
            self.detector.setScoreThreshold(0.40)
            _, faces = self.detector.detect(img)
            self.detector.setScoreThreshold(0.55)

        if faces is None or len(faces) == 0:
            return None, None, "No clear face detected in the uploaded image. Please upload a clear front-facing photo."

        # Take the most prominent face (highest confidence or largest area)
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
        force_recognition: bool = False
    ) -> Dict[str, Any]:
        """
        Main pipeline:
        1. Check motion.
        2. If motion detected or force_recognition: detect faces and recognize against known persons.
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
                    if score < 0.50:
                        continue

                    # Align and extract embedding
                    aligned_face = self.recognizer.alignCrop(frame_bgr, face)
                    current_feat = self.recognizer.feature(aligned_face)
                    current_feat = current_feat / np.linalg.norm(current_feat)
                    curr_emb = current_feat.flatten().tolist()

                    best_match = None
                    best_sim = -1.0

                    for person in known_persons:
                        emb = person.get("embedding")
                        if emb and len(emb) == 128:
                            sim = self.compare_embeddings(curr_emb, emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_match = person

                    is_matched = best_sim >= COSINE_THRESHOLD and best_match is not None

                    # Calculate display confidence (scaled 0-100%)
                    if is_matched:
                        # scale from [COSINE_THRESHOLD, 0.85] to [70%, 99%]
                        confidence = min(99.9, 70.0 + (best_sim - COSINE_THRESHOLD) * 75.0)
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
                        "confidence": round(confidence, 1)
                    })

        return {
            "motion_detected": motion_detected,
            "motion_score": motion_score,
            "faces": faces_results
        }
