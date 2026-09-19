"""Webcam capture and pretrained person detection."""

import time
import threading
from collections import deque
from threading import Lock

import cv2
from action_model import ViolenceActionRecognizer

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class VideoProcessor:
    """Run pretrained COCO person detection and ByteTrack on webcam frames."""

    def __init__(self, confidence_threshold=0.50, camera_index=0):
        if YOLO is None:
            raise RuntimeError(
                "Person detection requires ultralytics. Run run.bat again or "
                "install the packages in requirements.txt."
            )
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.camera_index = camera_index
        # Downloads the small pretrained model once, then reuses its local cache.
        self.detector = YOLO("yolo11n.pt")
        self.action_recognizer = ViolenceActionRecognizer()
        self.cap = None
        self.running = False
        self.frame = None
        self.persons = []
        self.active_track_ids = []
        self.action_frames = deque(maxlen=self.action_recognizer.sequence_length)
        self.action_confidence = 0.0
        self.action_confidence_history = deque(maxlen=5)
        self.is_violence_detected = False
        self.last_alert_id = 0
        self.frame_count = 0
        self.inference_fps = 0.0
        self.lock = Lock()
        self.alerts = deque(maxlen=100)  # Retained for API compatibility.

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        threading.Thread(target=self._process_frames, daemon=True).start()

    def _process_frames(self):
        while self.running:
            started_at = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            # COCO class 0 is "person". ByteTrack compares consecutive frames
            # and assigns an ID that persists while the person remains visible.
            result = self.detector.track(
                frame,
                classes=[0],
                conf=self.confidence_threshold,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            persons = []
            track_ids = result.boxes.id
            for index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                persons.append({
                    "track_id": int(track_ids[index].item()) if track_ids is not None else None,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(box.conf[0]), 4),
                })
            self.action_frames.append(frame.copy())
            # Video classification is much more expensive than person detection.
            # Run it on a rolling clip approximately every 16 processed frames.
            if (self.action_recognizer.ready
                    and len(self.action_frames) == self.action_recognizer.sequence_length
                    and (self.frame_count + 1) % 16 == 0):
                self.action_confidence_history.append(
                    self.action_recognizer.predict(list(self.action_frames))
                )
                self.action_confidence = float(sum(self.action_confidence_history) / len(self.action_confidence_history))
            was_violence_detected = self.is_violence_detected
            self.is_violence_detected = (
                self.action_recognizer.ready
                and self.action_confidence >= self.action_recognizer.threshold
            )
            if self.is_violence_detected and not was_violence_detected:
                self.last_alert_id += 1
                self.alerts.append({
                    "id": self.last_alert_id,
                    "timestamp": time.time(),
                    "confidence": round(self.action_confidence, 4),
                    "frame_number": self.frame_count + 1,
                    "status": "ACTIVE",
                    "type": "POSSIBLE_VIOLENCE",
                })
            elapsed = max(time.perf_counter() - started_at, 0.001)
            with self.lock:
                self.frame = frame.copy()
                self.persons = persons
                self.active_track_ids = sorted(
                    person["track_id"] for person in persons
                    if person["track_id"] is not None
                )
                self.frame_count += 1
                self.inference_fps = round(1 / elapsed, 1)

    def get_frame_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
            persons = list(self.persons)
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"Person #{person['track_id'] if person['track_id'] is not None else '?'} "
                f"{person['confidence']:.0%}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2,
            )
        cv2.putText(frame, f"People detected: {len(persons)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        with self.lock:
            action_ready = self.action_recognizer.ready
            action_confidence = self.action_confidence
            violence_detected = self.is_violence_detected
        if action_ready:
            color = (0, 0, 255) if violence_detected else (0, 180, 255)
            cv2.putText(frame, f"Possible violence: {action_confidence:.0%}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def get_data(self):
        with self.lock:
            persons = list(self.persons)
            return {
                "human_detected": bool(persons),
                "person_count": len(persons),
                "persons": persons,
                "active_track_ids": list(self.active_track_ids),
                "confidence_threshold": self.confidence_threshold,
                "frame_count": self.frame_count,
                "inference_fps": self.inference_fps,
                "model": "YOLO11n (COCO pretrained)",
                "action_model_ready": self.action_recognizer.ready,
                "action_model": "MobileNetV3 + LSTM" if self.action_recognizer.ready else None,
                "violence_confidence": round(self.action_confidence, 4),
                "violence_threshold": self.action_recognizer.threshold,
                "is_violence_detected": self.is_violence_detected,
                # Compatibility field for clients that display a single confidence.
                "confidence": max((person["confidence"] for person in persons), default=0.0),
                "alerts_count": len(self.alerts),
            }

    def get_alerts(self, limit=10):
        return list(reversed(list(self.alerts)))[:limit]

    def set_threshold(self, threshold):
        self.confidence_threshold = max(0.0, min(1.0, float(threshold)))

    def acknowledge_alert(self, alert_id):
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["status"] = "ACKNOWLEDGED"
                return True
        return False

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
