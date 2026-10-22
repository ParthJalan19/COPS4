import os
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "cops_new")
USE_MOCK_DB = os.getenv("USE_MOCK_DB", "false").lower() in ("true", "1", "yes")

_db_client = None
_db = None
_db_status = {"connected": False, "provider": "uninitialized", "message": "", "db_name": MONGODB_DB_NAME}
_local_db_file = "cops_new_local.json"


def _init_local_db():
    if not os.path.exists(_local_db_file):
        with open(_local_db_file, "w", encoding="utf-8") as f:
            json.dump({"watchlist": [], "alerts": [], "cameras": []}, f, indent=2)


def _load_local_db() -> Dict[str, list]:
    _init_local_db()
    try:
        with open(_local_db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("watchlist", [])
            data.setdefault("alerts", [])
            data.setdefault("cameras", [])
            return data
    except Exception:
        return {"watchlist": [], "alerts": [], "cameras": []}


def _save_local_db(data: Dict[str, list]):
    with open(_local_db_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def init_db():
    global _db_client, _db, _db_status, MONGODB_URI, MONGODB_DB_NAME, USE_MOCK_DB
    load_dotenv(override=True)
    MONGODB_URI = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "cops_new")
    USE_MOCK_DB = os.getenv("USE_MOCK_DB", "false").lower() in ("true", "1", "yes")

    if USE_MOCK_DB:
        _init_local_db()
        _db_status = {
            "connected": True,
            "provider": f"Local Fallback / Mock ({MONGODB_DB_NAME})",
            "db_name": MONGODB_DB_NAME,
            "message": f"Running locally for database '{MONGODB_DB_NAME}'"
        }
        return

    try:
        import pymongo
        uris_to_try = [MONGODB_URI]
        if "#" in MONGODB_URI and "%23" not in MONGODB_URI:
            uris_to_try.append(MONGODB_URI.replace("#", "%23"))

        last_err = None
        client = None
        for test_uri in uris_to_try:
            if not test_uri:
                continue
            try:
                print(f"[Database] Connecting to MongoDB Atlas ({MONGODB_DB_NAME})...")
                c = pymongo.MongoClient(test_uri, serverSelectionTimeoutMS=4000)
                c.admin.command('ping')
                client = c
                break
            except Exception as e:
                last_err = e

        if client is None:
            raise last_err or Exception("Could not connect to MongoDB Atlas")

        _db_client = client
        _db = client[MONGODB_DB_NAME]
        _db_status = {
            "connected": True,
            "provider": "MongoDB Atlas",
            "db_name": MONGODB_DB_NAME,
            "message": f"Connected to MongoDB Atlas database '{MONGODB_DB_NAME}', using 'watchlist' table"
        }
        print(f"[Database] {_db_status['message']}")
    except Exception as e:
        err_msg = str(e)
        print(f"[Database] MongoDB Atlas connection failed: {err_msg}")
        _init_local_db()
        _db_status = {
            "connected": False,
            "provider": f"Local Table ({MONGODB_DB_NAME})",
            "db_name": MONGODB_DB_NAME,
            "message": "Atlas authentication failed. Running on local table so watchlist and alerts work seamlessly."
        }


# Initialize on import
init_db()


def get_db_status() -> Dict[str, Any]:
    global _db_status
    if not _db_status.get("connected"):
        init_db()
    return _db_status


# ---------------- WATCHLIST TABLE ----------------
def add_to_watchlist(name: str, info: str, embedding: List[float], image_data: str) -> Dict[str, Any]:
    """Adds a new player/person to the watchlist table."""
    record = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "info": info.strip(),
        "embedding": embedding,
        "image": image_data,
        "status": "WATCHED",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if _db is not None:
        try:
            doc = record.copy()
            doc["_id"] = doc["id"]
            _db.watchlist.insert_one(doc)
            return record
        except Exception as e:
            print(f"[Database] Error inserting into MongoDB Atlas watchlist: {e}")

    # Fallback to local table
    data = _load_local_db()
    data["watchlist"].append(record)
    _save_local_db(data)
    return record


def get_watchlist() -> List[Dict[str, Any]]:
    """Retrieves all players from the watchlist table."""
    if _db is not None:
        try:
            docs = list(_db.watchlist.find())
            results = []
            for d in docs:
                item = {
                    "id": str(d.get("_id", d.get("id"))),
                    "name": d.get("name", "Unknown Player"),
                    "info": d.get("info", ""),
                    "embedding": d.get("embedding", []),
                    "image": d.get("image", ""),
                    "status": d.get("status", "WATCHED"),
                    "created_at": d.get("created_at", "")
                }
                results.append(item)
            return results
        except Exception as e:
            print(f"[Database] Error reading from Atlas watchlist: {e}")

    data = _load_local_db()
    return data.get("watchlist", [])


def delete_from_watchlist(item_id: str) -> bool:
    """Deletes a player from the watchlist table."""
    if _db is not None:
        try:
            res = _db.watchlist.delete_one({"_id": item_id})
            if res.deleted_count == 0:
                _db.watchlist.delete_one({"id": item_id})
            return True
        except Exception as e:
            print(f"[Database] Error deleting from Atlas watchlist: {e}")

    data = _load_local_db()
    data["watchlist"] = [w for w in data.get("watchlist", []) if w.get("id") != item_id]
    _save_local_db(data)
    return True


# ---------------- CAMERAS TABLE (Surveillance Wall) ----------------
def save_camera(
    cam_id: str,
    name: str,
    cam_type: str,
    stream_url: Optional[str] = None,
    device_id: Optional[str] = None
) -> Dict[str, Any]:
    """Persists a camera feed configuration in MongoDB Atlas."""
    cam_record = {
        "id": cam_id,
        "name": name.strip(),
        "type": cam_type,  # 'webcam', 'mobile', 'cctv_ip'
        "stream_url": (stream_url or "").strip(),
        "device_id": (device_id or "").strip(),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if _db is not None:
        try:
            doc = cam_record.copy()
            doc["_id"] = doc["id"]
            _db.cameras.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            return cam_record
        except Exception as e:
            print(f"[Database] Error saving camera to Atlas: {e}")

    data = _load_local_db()
    data["cameras"] = [c for c in data.get("cameras", []) if c.get("id") != cam_id]
    data["cameras"].append(cam_record)
    _save_local_db(data)
    return cam_record


def get_cameras() -> List[Dict[str, Any]]:
    """Retrieves all registered cameras for the camera wall."""
    if _db is not None:
        try:
            docs = list(_db.cameras.find())
            results = []
            for d in docs:
                item = {
                    "id": str(d.get("_id", d.get("id"))),
                    "name": d.get("name", "Camera"),
                    "type": d.get("type", "webcam"),
                    "stream_url": d.get("stream_url", ""),
                    "device_id": d.get("device_id", ""),
                    "created_at": d.get("created_at", "")
                }
                results.append(item)
            return results
        except Exception as e:
            print(f"[Database] Error reading cameras from Atlas: {e}")

    data = _load_local_db()
    return data.get("cameras", [])


def delete_camera(cam_id: str) -> bool:
    """Removes a camera from the wall."""
    if _db is not None:
        try:
            _db.cameras.delete_one({"_id": cam_id})
            return True
        except Exception as e:
            print(f"[Database] Error deleting camera from Atlas: {e}")

    data = _load_local_db()
    data["cameras"] = [c for c in data.get("cameras", []) if c.get("id") != cam_id]
    _save_local_db(data)
    return True


# ---------------- ALERTS TABLE ----------------
def generate_alert(
    player_id: str,
    player_name: str,
    info: str,
    confidence: float,
    snapshot: str,
    camera_id: str = "cam_main",
    camera_name: str = "Main Surveillance"
) -> Dict[str, Any]:
    """Generates and logs an alert when a watched player is detected."""
    alert = {
        "id": str(uuid.uuid4()),
        "player_id": player_id,
        "player_name": player_name,
        "info": info,
        "confidence": round(confidence * 100, 1),
        "snapshot": snapshot,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "alert_type": "PLAYER_WATCHLIST_MATCH",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if _db is not None:
        try:
            doc = alert.copy()
            doc["_id"] = doc["id"]
            _db.alerts.insert_one(doc)
            return alert
        except Exception as e:
            print(f"[Database] Error inserting into Atlas alerts: {e}")

    data = _load_local_db()
    data["alerts"].insert(0, alert)
    data["alerts"] = data["alerts"][:60]
    _save_local_db(data)
    return alert


def generate_weapon_alert(
    weapon_class: str,
    label: str,
    confidence: float,
    snapshot: str,
    camera_id: str = "cam_main",
    camera_name: str = "Main Surveillance"
) -> Dict[str, Any]:
    """Generates and logs an emergency alert when a weapon (knife/gun) is detected."""
    alert = {
        "id": str(uuid.uuid4()),
        "player_id": "",
        "player_name": f"CRITICAL: {label.upper()}",
        "info": f"Armed Threat Detected: {label}",
        "confidence": round(confidence, 1),
        "snapshot": snapshot,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "alert_type": "WEAPON_DETECTED",
        "weapon_type": weapon_class,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if _db is not None:
        try:
            doc = alert.copy()
            doc["_id"] = doc["id"]
            _db.alerts.insert_one(doc)
            return alert
        except Exception as e:
            print(f"[Database] Error inserting weapon alert into Atlas: {e}")

    data = _load_local_db()
    data["alerts"].insert(0, alert)
    data["alerts"] = data["alerts"][:60]
    _save_local_db(data)
    return alert


def get_alerts(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieves generated alerts."""
    if _db is not None:
        try:
            docs = list(_db.alerts.find().sort("timestamp", -1).limit(limit))
            results = []
            for d in docs:
                item = {
                    "id": str(d.get("_id", d.get("id"))),
                    "player_id": d.get("player_id", ""),
                    "player_name": d.get("player_name", ""),
                    "info": d.get("info", ""),
                    "confidence": d.get("confidence", 0),
                    "snapshot": d.get("snapshot", ""),
                    "camera_id": d.get("camera_id", "cam_main"),
                    "camera_name": d.get("camera_name", "Main Camera"),
                    "alert_type": d.get("alert_type", "PLAYER_WATCHLIST_MATCH"),
                    "timestamp": d.get("timestamp", "")
                }
                results.append(item)
            return results
        except Exception as e:
            print(f"[Database] Error reading alerts from Atlas: {e}")

    data = _load_local_db()
    return data.get("alerts", [])[:limit]


def clear_alerts() -> bool:
    if _db is not None:
        try:
            _db.alerts.delete_many({})
            return True
        except Exception as e:
            print(f"[Database] Error clearing alerts in Atlas: {e}")

    data = _load_local_db()
    data["alerts"] = []
    _save_local_db(data)
    return True
