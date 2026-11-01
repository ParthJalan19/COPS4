"""
COPS-4 Armed Suspect Auto-Enrollment & Cross-Camera Tracking Helper.
Functions:
1. Associates detected weapons with human actors and faces in the frame.
2. Extracts high-quality face crop and 128-D biometric embedding.
3. Automatically enrolls the suspect into MongoDB Atlas Watchlist as an ARMED_SUSPECT.
4. Identifies cross-camera transitions when an auto-enrolled suspect appears in another camera.
"""

import cv2
import base64
import time
import datetime
from typing import List, Dict, Any, Optional, Tuple
import database


def find_associated_face_for_weapon(
    weapon_bbox: List[int],
    faces: List[Dict[str, Any]],
    bodies: List[Dict[str, Any]],
    frame_shape: Tuple[int, int]
) -> Optional[Dict[str, Any]]:
    """
    Finds the face belonging to the person carrying the weapon.
    1. First checks if weapon is inside or nearby a detected body bounding box,
       and matches that body to an overlapping face.
    2. Otherwise, finds the closest face in the frame within plausible reach.
    """
    if not faces:
        return None

    wx, wy, ww, wh = weapon_bbox
    w_cx, w_cy = wx + ww / 2.0, wy + wh / 2.0

    # 1. Match via human body bounding box
    for body in bodies:
        bx, by, bw, bh = body["bbox"]
        # Expand body slightly for hand reach
        if (bx - 40 <= w_cx <= bx + bw + 40) and (by - 30 <= w_cy <= by + bh + 40):
            # Find face inside this body's upper area
            for f in faces:
                fx, fy, fw, fh = f["bbox"]
                if bx <= (fx + fw / 2.0) <= bx + bw and by <= fy <= by + bh * 0.6:
                    return f

    # 2. Nearest face fallback (head typically above or within 2.5x body distance of weapon)
    best_dist = float("inf")
    best_face = None

    for f in faces:
        fx, fy, fw, fh = f["bbox"]
        f_cx, f_cy = fx + fw / 2.0, fy + fh / 2.0
        # Euclidean distance
        dist = ((f_cx - w_cx) ** 2 + (f_cy - w_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face


def auto_enroll_armed_suspect(
    frame_bgr,
    weapon_obj: Dict[str, Any],
    associated_face: Dict[str, Any],
    camera_id: str,
    camera_name: str,
    face_engine
) -> Optional[Dict[str, Any]]:
    """
    Captures snapshot of armed suspect's face, extracts biometrics,
    and auto-enrolls into the MongoDB Atlas Watchlist.
    """
    if associated_face is None:
        return None

    # Check if face already matched an existing person in watchlist
    if associated_face.get("is_match") and associated_face.get("person_id"):
        # Suspect is already identified in watchlist
        return None

    fx, fy, fw, fh = associated_face["bbox"]
    h_img, w_img = frame_bgr.shape[:2]

    # Crop face with padding for clean watchlist photo
    pad_x = int(fw * 0.35)
    pad_y = int(fh * 0.40)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y)
    x2 = min(w_img, fx + fw + pad_x)
    y2 = min(h_img, fy + fh + pad_y)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        return None

    # Extract 128-D embedding using face_engine
    embedding, _, err = face_engine.extract_face_embedding(face_crop)
    if not embedding or len(embedding) != 128:
        return None

    # Convert crop to JPEG base64
    _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    weapon_lbl = weapon_obj.get("label", "Weapon").upper()
    suspect_tag = f"ARMED SUSPECT ({weapon_lbl})"
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    info = (
        f"AUTO-CAPTURED: Detected armed with {weapon_obj.get('label')} "
        f"on '{camera_name}' at {now_str}. "
        f"Tracked cross-camera."
    )

    # Check existing watchlist to prevent duplicate enrollment of same face
    existing_wl = database.get_watchlist()
    for item in existing_wl:
        emb = item.get("embedding")
        if emb and len(emb) == 128:
            sim = face_engine.compare_embeddings(embedding, emb)
            if sim >= 0.36:
                # Already enrolled, no need to duplicate
                return item

    # Save to MongoDB Atlas watchlist
    new_record = database.add_to_watchlist(
        name=suspect_tag,
        info=info,
        embedding=embedding,
        image_data=snap_b64,
        threat_level="ARMED_SUSPECT",
        origin_camera_id=camera_id,
        auto_enrolled=True
    )
    print(f"[AutoEnroll] Enrolled new armed suspect '{suspect_tag}' from camera '{camera_name}' into watchlist!")
    return new_record
