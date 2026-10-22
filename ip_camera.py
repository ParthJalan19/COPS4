import time
import threading
import cv2
import numpy as np
from typing import Dict, Optional, Generator

class StreamManager:
    def __init__(self):
        # Maps camera_id -> {"cap": cv2.VideoCapture, "latest_frame": np.ndarray, "thread": Thread, "running": bool, "url": str}
        self.streams: Dict[str, dict] = {}
        # Stores frames pushed from mobile phones or browser clients: camera_id -> (timestamp, frame_bgr)
        self.mobile_frames: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def add_cctv_stream(self, camera_id: str, stream_url: str):
        """Starts a background worker thread to pull frames from a CCTV RTSP or HTTP MJPEG stream."""
        self.remove_stream(camera_id)

        stream_info = {
            "url": stream_url,
            "running": True,
            "latest_frame": None,
            "last_active": time.time()
        }

        def worker():
            print(f"[CCTV] Starting connection to {stream_url} for camera {camera_id}...")
            cap = cv2.VideoCapture(stream_url)
            # Reduce buffer size for lower latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fail_count = 0
            while stream_info["running"]:
                ret, frame = cap.read()
                if ret and frame is not None:
                    fail_count = 0
                    with self.lock:
                        stream_info["latest_frame"] = frame
                        stream_info["last_active"] = time.time()
                    time.sleep(0.03)  # ~30 FPS
                else:
                    fail_count += 1
                    time.sleep(1.0)
                    if fail_count > 5:
                        print(f"[CCTV] Reconnecting to {stream_url}...")
                        cap.release()
                        cap = cv2.VideoCapture(stream_url)
                        fail_count = 0

            cap.release()
            print(f"[CCTV] Stopped stream for camera {camera_id}")

        t = threading.Thread(target=worker, daemon=True)
        stream_info["thread"] = t
        with self.lock:
            self.streams[camera_id] = stream_info
        t.start()

    def remove_stream(self, camera_id: str):
        with self.lock:
            if camera_id in self.streams:
                self.streams[camera_id]["running"] = False
                del self.streams[camera_id]
            if camera_id in self.mobile_frames:
                del self.mobile_frames[camera_id]

    def set_mobile_frame(self, camera_id: str, frame_bgr: np.ndarray):
        with self.lock:
            self.mobile_frames[camera_id] = {
                "latest_frame": frame_bgr,
                "timestamp": time.time()
            }

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        with self.lock:
            if camera_id in self.mobile_frames:
                # If frame is less than 5 seconds old, return it
                info = self.mobile_frames[camera_id]
                if time.time() - info["timestamp"] < 6.0:
                    return info["latest_frame"]

            if camera_id in self.streams:
                return self.streams[camera_id].get("latest_frame")

        return None

    def generate_mjpeg_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """Yields MJPEG multipart stream for <img> elements in the browser."""
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Connecting to Stream...", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        _, blank_buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
        blank_bytes = blank_buf.tobytes()

        while True:
            frame = self.get_latest_frame(camera_id)
            if frame is not None:
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            else:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')

            time.sleep(0.08)  # ~12 FPS for smooth network streaming


stream_manager = StreamManager()
