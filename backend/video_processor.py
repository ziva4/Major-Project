import cv2
import numpy as np
import threading
import time
from collections import deque
from train_model import ViolenceDetectionModel

class VideoProcessor:
    def __init__(self, confidence_threshold=0.6):
        self.detector = ViolenceDetectionModel()
        self.detector.load_model()

        self.cap = None
        self.running = False
        self.frame = None
        self.confidence = 0
        self.is_violence_detected = False
        self.confidence_threshold = confidence_threshold

        # Alert history
        self.alerts = deque(maxlen=100)
        self.frame_count = 0
        self.human_detected = False

        # Smoothing confidence over frames
        self.confidence_history = deque(maxlen=5)

    def start(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        thread = threading.Thread(target=self._process_frames, daemon=True)
        thread.start()

    def _process_frames(self):
        """Process video frames in background"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.frame_count += 1

            # Get violence confidence
            violence_confidence = self.detector.predict(frame)
            self.confidence_history.append(violence_confidence)

            # Smooth confidence
            smoothed_confidence = np.mean(list(self.confidence_history))
            self.confidence = smoothed_confidence

            # Check if violence detected
            was_detected = self.is_violence_detected
            self.is_violence_detected = self.confidence > self.confidence_threshold

            # Create alert on detection
            if self.is_violence_detected and not was_detected:
                alert = {
                    'id': len(self.alerts),
                    'timestamp': time.time(),
                    'confidence': float(self.confidence),
                    'frame_number': self.frame_count,
                    'status': 'ACTIVE'
                }
                self.alerts.append(alert)
                print(f"🚨 VIOLENCE DETECTED - Confidence: {self.confidence:.2%}")

            # Store frame for serving to frontend
            self.frame = frame.copy()

            time.sleep(0.01)

    def get_frame_jpeg(self):
        """Get current frame as JPEG bytes"""
        if self.frame is None:
            return None

        # Add overlay with confidence
        frame = self.frame.copy()
        frame = cv2.putText(
            frame, f"Confidence: {self.confidence:.2%}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if self.confidence < self.confidence_threshold else (0, 0, 255),
            2
        )

        # Add status
        status = "⚠️ VIOLENCE DETECTED" if self.is_violence_detected else "Normal"
        color = (0, 0, 255) if self.is_violence_detected else (0, 255, 0)
        frame = cv2.putText(
            frame, status, (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if ret else None

    def get_data(self):
        """Get current detection data"""
        return {
            'confidence': round(self.confidence, 4),
            'is_violence_detected': self.is_violence_detected,
            'confidence_threshold': self.confidence_threshold,
            'frame_count': self.frame_count,
            'alerts_count': len(self.alerts)
        }

    def get_alerts(self, limit=10):
        """Get recent alerts"""
        return list(reversed(list(self.alerts)))[:limit]

    def set_threshold(self, threshold):
        """Set confidence threshold"""
        self.confidence_threshold = max(0, min(1, threshold))

    def stop(self):
        """Stop video capture"""
        self.running = False
        if self.cap:
            self.cap.release()

    def acknowledge_alert(self, alert_id):
        """Mark alert as acknowledged"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'ACKNOWLEDGED'
                break
