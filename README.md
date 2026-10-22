# COPS Multi-Camera Surveillance Wall & Face Recognition

A real-time multi-camera surveillance wall, face detection, recognition, and motion tracking web application built with **FastAPI**, **OpenCV YuNet + SFace Biometrics**, and **MongoDB Atlas**.

---

## What's New: Surveillance Camera Wall

- **Multi-Camera Grid**:
  - Dynamic **1x1 Focus Mode**, **2x2 Grid (4 Cams)**, and **3x3 Grid (9 Cams)**.
  - Fullscreen focus on any camera tile with a single click.
- **Multiple Camera Sources**:
  1. **Laptop & USB Webcams**: Connects directly to internal webcams and USB cameras attached to the computer.
  2. **Mobile Phone Cameras (Wireless)**: Turn any smartphone (iPhone/Android) into a wireless camera node by scanning the on-screen QR code or opening `http://<your-ip>:8000/mobile`. Features rear/front camera switching and live video streaming.
  3. **CCTV / IP Camera Streams**: Connect RTSP or HTTP MJPEG camera streams (e.g., CCTV DVR, Dahua, Hikvision, or the mobile app *IP Webcam*).
- **Independent Facial Recognition per Camera**:
  - Each camera stream runs independent face detection and movement tracking.
  - When a watched target is identified on any camera, that specific camera tile halts to display the target while other cameras continue monitoring.
  - The alert log records the exact camera source (e.g. `Main Entrance - CCTV`, `Patrol Phone Cam`, `Laptop Webcam`).

---

## Database & Collections (MongoDB Atlas)

- **Database**: `cops_new`
- **Watchlist Table**: `watchlist` (registered targets with 128-d deep facial embeddings)
- **Generated Alerts Table**: `alerts` (detection logs with timestamp, camera name, and snapshot)
- **Cameras Table**: `cameras` (persisted camera feed configurations)

---

## How to Run

1. Start the server:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   Or double-click `start.bat`.

2. Open the main surveillance wall:
   👉 **[http://localhost:8000](http://localhost:8000)**

3. Connect a mobile phone camera:
   - Click **"+ Add Camera Feed"** → **Mobile Phone** tab.
   - Scan the QR code with your phone or open the link on your phone browser.
