import os
import io
import time
import base64
import socket
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

import database
from face_engine import FaceEngine
from weapon_detector import weapon_detector
from body_detector import BodyDetector
from ip_camera import stream_manager

app = FastAPI(title="COPS Camera Wall & Face Recognition System", version="3.0.0")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_engine = FaceEngine()
body_detector = BodyDetector()

# Throttle alerts per player/camera and weapon/camera
_last_alert_times = {}
_last_weapon_alert_times = {}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class DetectRequest(BaseModel):
    image: str
    force_recognition: bool = False
    camera_id: Optional[str] = "cam_main"
    camera_name: Optional[str] = "Main Camera"
    weapon_threshold: Optional[float] = 0.20


class MobileFrameRequest(BaseModel):
    camera_id: str
    camera_name: Optional[str] = "Mobile Phone Camera"
    image: str
    force_recognition: bool = True
    weapon_threshold: Optional[float] = 0.20


class CameraConfigRequest(BaseModel):
    id: str
    name: str
    type: str  # 'webcam', 'mobile', 'cctv_ip'
    stream_url: Optional[str] = ""
    device_id: Optional[str] = ""


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


# Re-initialize any persisted CCTV streams on startup
def _restore_cctv_streams():
    cams = database.get_cameras()
    for cam in cams:
        if cam.get("type") == "cctv_ip" and cam.get("stream_url"):
            stream_manager.add_cctv_stream(cam["id"], cam["stream_url"])

_restore_cctv_streams()


@app.get("/api/status")
def get_status():
    db_status = database.get_db_status()
    watchlist = database.get_watchlist()
    alerts = database.get_alerts(limit=5)
    cameras = database.get_cameras()
    return {
        "status": "online",
        "database": db_status,
        "database_name": db_status.get("db_name", "cops_new"),
        "watchlist_table": "watchlist",
        "alerts_table": "alerts",
        "cameras_table": "cameras",
        "watchlist_count": len(watchlist),
        "recent_alerts_count": len(alerts),
        "camera_count": len(cameras),
        "network_ip": get_local_ip(),
        "models_loaded": {
            "face_detector": "OpenCV YuNet ONNX (0.35 threshold, 5-pt Ocular/Cornea Landmarks)",
            "face_recognizer": "OpenCV SFace ONNX (Periocular Biometrics & Mask-Aware Matching)",
            "body_detector": "YOLOv8n Full Body Human Detector (person_detection_yolov8)",
            "weapon_detector": "Dual-Model YOLOv8 Ensemble (Letterbox Preprocessed, Guns, Pistols, Knives, Blades)",
            "motion_detector": "Frame Differencing & Contour Area"
        }
    }


@app.get("/api/network-ip")
def get_network_ip_endpoint():
    ip = get_local_ip()
    port = int(os.getenv("PORT", 8000))
    return {
        "ip": ip,
        "port": port,
        "mobile_url": f"http://{ip}:{port}/mobile"
    }


# ---------------- CAMERA MANAGEMENT (CAMERA WALL) ----------------
@app.get("/api/cameras")
def list_cameras():
    return database.get_cameras()


@app.post("/api/cameras")
def add_camera_endpoint(req: CameraConfigRequest):
    cam = database.save_camera(
        cam_id=req.id,
        name=req.name,
        cam_type=req.type,
        stream_url=req.stream_url,
        device_id=req.device_id
    )
    if req.type == "cctv_ip" and req.stream_url:
        stream_manager.add_cctv_stream(req.id, req.stream_url)

    return {"success": True, "camera": cam}


@app.delete("/api/cameras/{camera_id}")
def delete_camera_endpoint(camera_id: str):
    stream_manager.remove_stream(camera_id)
    success = database.delete_camera(camera_id)
    return {"success": success}


@app.get("/api/stream/{camera_id}")
def stream_camera(camera_id: str):
    """Streams live MJPEG frames for CCTV / IP cameras."""
    return StreamingResponse(
        stream_manager.generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- MOBILE CAMERA STREAM ----------------
@app.post("/api/mobile/frame")
def mobile_frame_endpoint(req: MobileFrameRequest):
    img_data = req.image
    if not img_data:
        raise HTTPException(status_code=400, detail="No image provided")

    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    # Store frame in StreamManager so the camera wall tile can display it
    stream_manager.set_mobile_frame(req.camera_id, frame)

    # Process recognition on frame
    watchlist_players = database.get_watchlist()
    result = face_engine.process_frame(
        frame_bgr=frame,
        known_persons=watchlist_players,
        force_recognition=req.force_recognition
    )

    # Process full-body human detection (head to toe)
    bodies = body_detector.detect_bodies(frame, conf_thresh=0.35)
    result["bodies"] = bodies
    result["has_body"] = len(bodies) > 0

    # Process weapon scanning (knives, guns/pistols)
    weapons = weapon_detector.detect_weapons(frame, custom_conf=req.weapon_threshold)
    result["weapons"] = weapons
    result["has_weapon"] = len(weapons) > 0

    now = time.time()
    generated_alerts = []

    # Process weapon emergency alerts
    for w_obj in weapons:
        w_cls = w_obj["class"]
        w_key = f"weapon_{req.camera_id}_{w_cls}"
        last_w_alert = _last_weapon_alert_times.get(w_key, 0)
        if now - last_w_alert > 3.0:
            _last_weapon_alert_times[w_key] = now
            wx, wy, ww, wh = w_obj["bbox"]
            fh, fw = frame.shape[:2]
            pad_x = int(ww * 0.2)
            pad_y = int(wh * 0.2)
            crop = frame[max(0, wy - pad_y):min(fh, wy + wh + pad_y), max(0, wx - pad_x):min(fw, wx + ww + pad_x)]
            snap_b64 = ""
            if crop.size > 0:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

            w_alert = database.generate_weapon_alert(
                weapon_class=w_cls,
                label=w_obj["label"],
                confidence=w_obj["confidence"],
                snapshot=snap_b64,
                camera_id=req.camera_id,
                camera_name=req.camera_name or "Mobile Phone Camera"
            )
            generated_alerts.append(w_alert)

    # Process watchlist target alerts (including masked suspects matched by ocular biometrics)
    for face in result.get("faces", []):
        if face.get("is_match") and face.get("person_id"):
            pid = face["person_id"]
            key = f"{req.camera_id}_{pid}"
            last_alert = _last_alert_times.get(key, 0)
            if now - last_alert > 3.5:
                _last_alert_times[key] = now
                x, y, w, h = face["bbox"]
                fh, fw = frame.shape[:2]
                pad_x = int(w * 0.25)
                pad_y = int(h * 0.25)
                crop = frame[max(0, y - pad_y):min(fh, y + h + pad_y), max(0, x - pad_x):min(fw, x + w + pad_x)]
                snap_b64 = ""
                if crop.size > 0:
                    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    snap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

                is_masked = face.get("is_masked", False)
                p_name = f"{face['name']} [😷 MASKED SUSPECT]" if is_masked else face["name"]
                info_text = f"👁️ Ocular / Cornea Match ({face['confidence']}%) - Mask Detected" if is_masked else (face.get("info", "") or "Facial Recognition Match")

                alert_record = database.generate_alert(
                    player_id=pid,
                    player_name=p_name,
                    info=info_text,
                    confidence=face["confidence"] / 100.0,
                    snapshot=snap_b64,
                    camera_id=req.camera_id,
                    camera_name=req.camera_name or "Mobile Phone Camera"
                )
                generated_alerts.append(alert_record)

    result["alerts_generated"] = generated_alerts
    result["has_alert"] = len(generated_alerts) > 0
    return result


# Standby image buffer for mobile cameras when waiting for phone connection
_standby_img = np.full((360, 480, 3), (16, 21, 28), dtype=np.uint8)
cv2.circle(_standby_img, (240, 130), 45, (40, 50, 65), -1)
cv2.putText(_standby_img, "MOBILE CAMERA STANDBY", (95, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (88, 166, 255), 2)
cv2.putText(_standby_img, "Waiting for phone stream to connect...", (75, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 148, 158), 1)
_, _sbuf = cv2.imencode(".jpg", _standby_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
_STANDBY_BYTES = _sbuf.tobytes()

@app.get("/api/mobile/frame/{camera_id}")
def get_mobile_frame_preview(camera_id: str):
    """Returns the latest snapshot of a mobile camera for wall rendering."""
    frame = stream_manager.get_latest_frame(camera_id)
    if frame is None:
        return Response(content=_STANDBY_BYTES, media_type="image/jpeg", status_code=200)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ---------------- WATCHLIST ENDPOINTS ----------------
@app.get("/api/watchlist")
def list_watchlist():
    items = database.get_watchlist()
    cleaned = []
    for p in items:
        cleaned.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "info": p.get("info", p.get("details", "")),
            "image": p.get("image"),
            "status": p.get("status", "WATCHED"),
            "created_at": p.get("created_at"),
            "has_embedding": bool(p.get("embedding"))
        })
    return cleaned


@app.get("/api/persons")
def list_persons_alias():
    return list_watchlist()


@app.post("/api/watchlist")
async def add_watchlist_endpoint(
    name: str = Form(...),
    info: str = Form(""),
    file: UploadFile = File(...)
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Player name is required")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded photo is empty")

    embedding, bbox, err = face_engine.extract_face_embedding(contents)
    if err or embedding is None:
        raise HTTPException(status_code=400, detail=err or "Face extraction failed")

    img_b64 = "data:image/jpeg;base64," + base64.b64encode(contents).decode("utf-8")

    record = database.add_to_watchlist(
        name=name.strip(),
        info=info.strip(),
        embedding=embedding,
        image_data=img_b64
    )

    return {
        "success": True,
        "message": f"Player '{name}' added to watchlist table successfully in database 'cops_new'",
        "player": {
            "id": record["id"],
            "name": record["name"],
            "info": record["info"],
            "created_at": record["created_at"]
        }
    }


@app.post("/api/persons")
async def add_persons_alias(name: str = Form(...), details: str = Form(""), file: UploadFile = File(...)):
    return await add_watchlist_endpoint(name=name, info=details, file=file)


@app.delete("/api/watchlist/{item_id}")
def delete_watchlist_endpoint(item_id: str):
    success = database.delete_from_watchlist(item_id)
    return {"success": success, "message": "Player removed from watchlist table"}


@app.delete("/api/persons/{person_id}")
def delete_persons_alias(person_id: str):
    return delete_watchlist_endpoint(person_id)


# ---------------- DETECTION & ALERT GENERATION ----------------
@app.post("/api/detect")
def detect_endpoint(req: DetectRequest):
    img_data = req.image
    if not img_data:
        raise HTTPException(status_code=400, detail="No image provided")

    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    watchlist_players = database.get_watchlist()
    players_by_id = {p["id"]: p for p in watchlist_players}

    result = face_engine.process_frame(
        frame_bgr=frame,
        known_persons=watchlist_players,
        force_recognition=req.force_recognition
    )

    # Process full-body human detection (head to toe)
    bodies = body_detector.detect_bodies(frame, conf_thresh=0.35)
    result["bodies"] = bodies
    result["has_body"] = len(bodies) > 0

    # Process weapon scanning (knives, guns/pistols)
    weapons = weapon_detector.detect_weapons(frame, custom_conf=req.weapon_threshold)
    result["weapons"] = weapons
    result["has_weapon"] = len(weapons) > 0

    now = time.time()
    generated_alerts = []

    # Process weapon emergency alerts
    for w_obj in weapons:
        w_cls = w_obj["class"]
        w_key = f"weapon_{req.camera_id or 'cam_main'}_{w_cls}"
        last_w_alert = _last_weapon_alert_times.get(w_key, 0)
        if now - last_w_alert > 3.0:
            _last_weapon_alert_times[w_key] = now
            wx, wy, ww, wh = w_obj["bbox"]
            fh, fw = frame.shape[:2]
            pad_x = int(ww * 0.2)
            pad_y = int(wh * 0.2)
            crop = frame[max(0, wy - pad_y):min(fh, wy + wh + pad_y), max(0, wx - pad_x):min(fw, wx + ww + pad_x)]
            snap_b64 = ""
            if crop.size > 0:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

            w_alert = database.generate_weapon_alert(
                weapon_class=w_cls,
                label=w_obj["label"],
                confidence=w_obj["confidence"],
                snapshot=snap_b64,
                camera_id=req.camera_id or "cam_main",
                camera_name=req.camera_name or "Main Surveillance"
            )
            generated_alerts.append(w_alert)

    # Process watchlist target alerts (including masked suspects matched by ocular biometrics)
    for face in result.get("faces", []):
        if face.get("is_match") and face.get("person_id"):
            pid = face["person_id"]
            player_obj = players_by_id.get(pid, {})
            face["registered_image"] = player_obj.get("image", "")

            x, y, w, h = face["bbox"]
            fh, fw = frame.shape[:2]
            pad_x = int(w * 0.25)
            pad_y = int(h * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x)
            y2 = min(fh, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]

            snap_b64 = ""
            if crop.size > 0:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snap_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")
            face["snapshot"] = snap_b64

            key = f"{req.camera_id or 'cam_main'}_{pid}"
            last_alert = _last_alert_times.get(key, 0)
            if now - last_alert > 3.0:
                _last_alert_times[key] = now
                is_masked = face.get("is_masked", False)
                p_name = f"{face['name']} [😷 MASKED SUSPECT]" if is_masked else face["name"]
                info_text = f"👁️ Ocular / Cornea Match ({face['confidence']}%) - Mask Detected" if is_masked else (face.get("info", face.get("details", "")) or "Facial Recognition Match")

                alert_record = database.generate_alert(
                    player_id=pid,
                    player_name=p_name,
                    info=info_text,
                    confidence=face["confidence"] / 100.0,
                    snapshot=snap_b64,
                    camera_id=req.camera_id or "cam_main",
                    camera_name=req.camera_name or "Main Surveillance"
                )
                generated_alerts.append(alert_record)

    result["alerts_generated"] = generated_alerts
    result["has_alert"] = len(generated_alerts) > 0
    return result


# ---------------- ALERTS ENDPOINTS ----------------
@app.get("/api/alerts")
def get_alerts_endpoint(limit: int = Query(25, ge=1, le=100)):
    return database.get_alerts(limit=limit)


@app.get("/api/detections")
def get_detections_alias(limit: int = Query(25, ge=1, le=100)):
    return database.get_alerts(limit=limit)


@app.delete("/api/alerts")
def clear_alerts_endpoint():
    database.clear_alerts()
    return {"success": True, "message": "Alerts cleared"}


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/mobile")
def serve_mobile_page():
    mobile_file = os.path.join(STATIC_DIR, "mobile.html")
    if os.path.exists(mobile_file):
        return FileResponse(mobile_file)
    return JSONResponse({"message": "Mobile streaming page not found"})


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Frontend not ready yet"})
