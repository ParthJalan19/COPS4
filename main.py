import os
import io
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

import database
from face_engine import FaceEngine

app = FastAPI(title="COPS Watchlist & Face Detection System", version="2.0.0")

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

# Throttle alerts so the same player isn't spamming alerts every 200ms
_last_alert_times = {}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class DetectRequest(BaseModel):
    image: str
    force_recognition: bool = False


@app.get("/api/status")
def get_status():
    db_status = database.get_db_status()
    watchlist = database.get_watchlist()
    alerts = database.get_alerts(limit=5)
    return {
        "status": "online",
        "database": db_status,
        "database_name": db_status.get("db_name", "cops_new"),
        "watchlist_table": "watchlist",
        "alerts_table": "alerts",
        "watchlist_count": len(watchlist),
        "recent_alerts_count": len(alerts),
        "models_loaded": {
            "face_detector": "OpenCV YuNet ONNX (face_detection_yunet)",
            "face_recognizer": "OpenCV SFace ONNX (face_recognition_sface)",
            "motion_detector": "Frame Differencing & Contour Area"
        }
    }


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


# Aliases for backwards compatibility
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

    # Extract face embedding
    embedding, bbox, err = face_engine.extract_face_embedding(contents)
    if err or embedding is None:
        raise HTTPException(status_code=400, detail=err or "Face extraction failed")

    # Convert to base64 preview
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(contents).decode("utf-8")

    # Save to MongoDB Atlas (cops_new.watchlist)
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

    # Fetch players from watchlist
    watchlist_players = database.get_watchlist()
    players_by_id = {p["id"]: p for p in watchlist_players}

    # Process frame
    result = face_engine.process_frame(
        frame_bgr=frame,
        known_persons=watchlist_players,
        force_recognition=req.force_recognition
    )

    now = time.time()
    generated_alerts = []

    # Check for matched players
    for face in result.get("faces", []):
        if face.get("is_match") and face.get("person_id"):
            pid = face["person_id"]
            player_obj = players_by_id.get(pid, {})
            face["registered_image"] = player_obj.get("image", "")

            # Capture snapshot thumbnail
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

            last_alert = _last_alert_times.get(pid, 0)
            if now - last_alert > 3.0:
                _last_alert_times[pid] = now
                alert_record = database.generate_alert(
                    player_id=pid,
                    player_name=face["name"],
                    info=face.get("info", face.get("details", "")),
                    confidence=face["confidence"] / 100.0,
                    snapshot=snap_b64
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


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Frontend not ready yet"})
