# COPS Watchlist Face Recognition & Alert System

A real-time face detection, recognition, and motion tracking web application built with **FastAPI**, **OpenCV YuNet + SFace Biometrics**, and **MongoDB Atlas**.

---

## Database & Tables

- **Database**: `cops_new`
- **Watchlist Table**: `watchlist`
  - Stores watched players/persons with name, alert details/info, face biometrics (128-d deep vector), and photo.
- **Generated Alerts Table**: `alerts`
  - Automatically records an alert whenever a watched player is detected by the camera upon movement.
  - Stores alert timestamp, detected player name, snapshot thumbnail, match confidence %, and alert notes.

---

## Features

1. **Watchlist Registration**:
   - Add any player/person to the `watchlist` table with their photo and info.
   - Extracts 128-dimensional deep face biometrics using OpenCV SFace ONNX.

2. **Movement & Player Detection**:
   - Live camera stream in browser.
   - **Motion Detection**: Tracks changes between frames using Gaussian difference and contour area analysis.
   - When motion is detected, scans faces and compares against the `watchlist` table.
   - **Instant Alert Generation**:
     - Flashes an emergency red alert banner on screen.
     - Plays an audio siren alert.
     - Logs the event and snapshot directly into the `alerts` table in database `cops_new`.

3. **MongoDB Atlas Configuration**:
   - Configured in `.env`:
     ```env
     MONGODB_URI="mongodb+srv://parthjalan18476_db_user:Parth%40!357@cluster0.sqtjclf.mongodb.net/cops_new?retryWrites=true&w=majority"
     MONGODB_DB_NAME="cops_new"
     USE_MOCK_DB=false
     ```
   - Automatically maintains a local fallback table if Atlas credentials fail authentication, ensuring 100% uptime.

---

## How to Run

1. Start the server:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   Or double-click `start.bat`.

2. Open your browser:
   ```
   http://127.0.0.1:8000
   ```
