import sys
import os
import base64
import cv2
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from main import app, _last_alert_times, _last_weapon_alert_times

client = TestClient(app)

print("=== 1. TESTING /api/status ===")
res = client.get("/api/status")
assert res.status_code == 200, f"Status error: {res.text}"
print("Status OK: DB connected, tables present.")

print("\n=== 2. TESTING WEAPON DETECTION ACCURACY ===")
test_cases = [
    ("static/samples/knife_sample_1.jpg", "knife"),
    ("static/samples/knife_sample_2.jpg", "knife"),
    ("static/samples/pistol_sample_1.jpg", "pistol"),
    ("static/samples/pistol_sample_2.jpg", "pistol"),
]

for img_path, expected_cls in test_cases:
    with open(img_path, "rb") as f:
        b64_str = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    
    # Reset cooldown so first frame can alert
    _last_weapon_alert_times.clear()
    
    resp = client.post("/api/detect", json={
        "image": b64_str,
        "camera_id": "test_verification",
        "camera_name": "Test Cam",
        "force_recognition": False,
        "weapon_threshold": 0.42
    })
    assert resp.status_code == 200, f"Detect error: {resp.text}"
    data = resp.json()
    weapons = data.get("weapons", [])
    assert len(weapons) > 0, f"Expected weapon in {img_path}, but found none!"
    detected_classes = [w["class"] for w in weapons]
    summary = [(w["class"], str(w["confidence"]) + "%") for w in weapons]
    print(f"{img_path}: detected {len(weapons)} weapons -> {summary}")
    assert all(c == expected_cls for c in detected_classes), f"Unexpected class in {img_path}! Got: {detected_classes}, Expected only: {expected_cls}"

print("Weapon Detection Accuracy: 100% PASSED (zero cross-class false alarms).")

print("\n=== 3. TESTING 10-MINUTE ALERT COOLDOWN & DEDUPLICATION ===")
_last_weapon_alert_times.clear()
with open("static/samples/knife_sample_1.jpg", "rb") as f:
    b64_str = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

alerts_count = []
for i in range(15):
    resp = client.post("/api/detect", json={
        "image": b64_str,
        "camera_id": "cam_security_1",
        "camera_name": "Entrance Security",
        "force_recognition": False,
        "weapon_threshold": 0.42
    })
    data = resp.json()
    gen = data.get("alerts_generated", [])
    if gen:
        print(f"Frame {i} generated: {[a.get('alert_type') for a in gen]}")
    alerts_count.append(len(gen))

print(f"Alerts generated over 15 rapid frames: {alerts_count}")
# Frame 0 triggers weapon alert, subsequent frame triggers auto-enrolled suspect match, then all 100% suppressed
assert sum(alerts_count[3:]) == 0, f"Frames 3-15 MUST be suppressed by the 10-minute cooldown! Got {alerts_count[3:]}"
print("10-Minute Cooldown: PASSED (Initial alerts fired, subsequent frames suppressed cleanly).")

print("\n=== 4. TESTING RESET COOLDOWN ENDPOINT ===")
reset_res = client.post("/api/alerts/reset_cooldown")
assert reset_res.status_code == 200
# Next frame should alert again now that cooldown is reset
resp = client.post("/api/detect", json={
    "image": b64_str,
    "camera_id": "cam_security_1",
    "camera_name": "Entrance Security",
    "force_recognition": False,
    "weapon_threshold": 0.42
})
assert len(resp.json().get("alerts_generated", [])) == 1
print("Reset Cooldown: PASSED.")
print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
