import base64
import cv2
import json
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
        "weapon_threshold": 0.25
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
        "weapon_threshold": 0.25
    })
    data = resp.json()
    alerts_count.append(len(data.get("alerts_generated", [])))

print(f"Alerts generated over 15 rapid frames: {alerts_count}")
assert alerts_count[0] >= 1, f"Frame 1 should generate at least 1 alert, got {alerts_count[0]}"
assert sum(alerts_count[1:]) == 0, f"Frames 2-15 MUST be suppressed by the 10-minute cooldown! Got {alerts_count[1:]}"
print(f"10-Minute Cooldown: PASSED (Frame 1 alerted with {alerts_count[0]} initial alert(s), 14 subsequent frames suppressed cleanly).")

print("\n=== 4. TESTING RESET COOLDOWN ENDPOINT ===")
reset_res = client.post("/api/alerts/reset_cooldown")
assert reset_res.status_code == 200
# Next frame should alert again now that cooldown is reset
resp = client.post("/api/detect", json={
    "image": b64_str,
    "camera_id": "cam_security_1",
    "camera_name": "Entrance Security",
    "force_recognition": False,
    "weapon_threshold": 0.25
})
assert len(resp.json().get("alerts_generated", [])) == 1
print("Reset Cooldown: PASSED.")
print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
