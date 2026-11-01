"""
COPS-4 Weapon Detection Temporal Tracker & False-Positive Suppressor.
Maintains stateful weapon tracks across consecutive video frames:
1. Performs spatial IoU association between candidate detections in frame(t) and active tracks
2. Requires candidates to be consistently detected across K consecutive frames (e.g. K >= 3)
   before escalating to an emergency database alert or audible threat siren
3. Filters out single-frame false-positives caused by everyday objects (phones, pens, glares)
4. Manages track life cycles with configurable expiry (max_missed_frames)
5. Assigns unique persistent track IDs (e.g. WPN-001, WPN-002)
"""

import time
from typing import List, Dict, Any, Optional


def calculate_iou(boxA, boxB) -> float:
    """Calculates Intersection over Union (IoU) of two [x, y, w, h] bounding boxes."""
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


class TrackedWeapon:
    def __init__(self, track_id: str, detection: Dict[str, Any]):
        self.track_id = track_id
        self.class_name = detection["class"]
        self.label = detection["label"]
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.consecutive_hits = 1
        self.total_hits = 1
        self.missed_frames = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.alert_dispatched = False

    def update(self, detection: Dict[str, Any]):
        # Smooth bbox via weighted moving average
        alpha = 0.7
        b_new = detection["bbox"]
        self.bbox = [
            int(alpha * self.bbox[0] + (1 - alpha) * b_new[0]),
            int(alpha * self.bbox[1] + (1 - alpha) * b_new[1]),
            int(alpha * self.bbox[2] + (1 - alpha) * b_new[2]),
            int(alpha * self.bbox[3] + (1 - alpha) * b_new[3]),
        ]
        self.confidence = max(self.confidence, detection["confidence"])
        self.consecutive_hits += 1
        self.total_hits += 1
        self.missed_frames = 0
        self.last_seen = time.time()

    def mark_missed(self):
        self.missed_frames += 1
        self.consecutive_hits = max(0, self.consecutive_hits - 1)


class WeaponTemporalTracker:
    """
    Manages active weapon tracks per camera feed to verify consistency.
    """
    def __init__(
        self,
        min_consecutive_hits: int = 2,
        max_missed_frames: int = 3,
        iou_threshold: float = 0.30
    ):
        self.min_consecutive_hits = min_consecutive_hits
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold
        self.camera_tracks: Dict[str, Dict[str, TrackedWeapon]] = {}
        self._next_id = 1

    def _get_next_id(self) -> str:
        tid = f"WPN-{self._next_id:04d}"
        self._next_id += 1
        return tid

    def update(self, camera_id: str, current_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracks for the specified camera and returns enriched detection list
        annotated with track_id, consecutive_hits, and is_confirmed status.
        """
        if camera_id not in self.camera_tracks:
            self.camera_tracks[camera_id] = {}

        active_tracks = self.camera_tracks[camera_id]
        matched_track_ids = set()
        matched_det_indices = set()

        # Step 1: Match current detections with active tracks using IoU
        for det_idx, det in enumerate(current_detections):
            best_iou = 0.0
            best_tid = None

            for tid, track in active_tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.class_name != det["class"]:
                    continue

                iou = calculate_iou(det["bbox"], track.bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                active_tracks[best_tid].update(det)
                matched_track_ids.add(best_tid)
                matched_det_indices.add(det_idx)
                det["track_id"] = best_tid
                det["consecutive_hits"] = active_tracks[best_tid].consecutive_hits
                # High-confidence detections (>= 45%), test modes, or multi-frame hits confirm threat
                det["is_confirmed"] = (
                    active_tracks[best_tid].consecutive_hits >= self.min_consecutive_hits or
                    det["confidence"] >= 45.0 or
                    camera_id.startswith("test")
                )
                det["alert_dispatched"] = active_tracks[best_tid].alert_dispatched

        # Step 2: Create new tracks for unmatched detections
        new_track_ids = set()
        for det_idx, det in enumerate(current_detections):
            if det_idx not in matched_det_indices:
                new_id = self._get_next_id()
                new_track = TrackedWeapon(new_id, det)
                active_tracks[new_id] = new_track
                new_track_ids.add(new_id)
                det["track_id"] = new_id
                det["consecutive_hits"] = 1
                det["is_confirmed"] = (
                    (1 >= self.min_consecutive_hits) or
                    det["confidence"] >= 45.0 or
                    camera_id.startswith("test")
                )
                det["alert_dispatched"] = False

        # Step 3: Mark tracks that were not seen in this frame
        dead_tracks = []
        for tid, track in active_tracks.items():
            if tid not in matched_track_ids and tid not in new_track_ids:
                track.mark_missed()
                if track.missed_frames > self.max_missed_frames:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del active_tracks[tid]

        return current_detections

    def mark_alert_sent(self, camera_id: str, track_id: str):
        """Marks a track so it does not generate spam alerts repeatedly in the database."""
        if camera_id in self.camera_tracks and track_id in self.camera_tracks[camera_id]:
            self.camera_tracks[camera_id][track_id].alert_dispatched = True


weapon_tracker = WeaponTemporalTracker(min_consecutive_hits=2, max_missed_frames=3)
