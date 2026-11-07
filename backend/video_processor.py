"""Person tracking plus asynchronous logistic classification of timed clips."""

import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, confidence_threshold=0.50, camera_index=0, *,
                 detector=None, recognizer=None, window_seconds=5.0,
                 action_interval=1.0, stale_seconds=3.0, alerts_enabled=False,
                 alert_threshold=0.70, load_action_model=True):
        if window_seconds <= 0 or action_interval <= 0 or stale_seconds <= 0:
            raise ValueError("Timing settings must be positive.")
        self.confidence_threshold = self._threshold(confidence_threshold)
        self.alert_threshold = self._threshold(alert_threshold)
        self.alerts_enabled = bool(alerts_enabled)
        self.camera_index = camera_index
        if detector is None:
            from ultralytics import YOLO
            detector = YOLO(str(Path(__file__).parent / "yolo11n.pt"))
        self.detector = detector
        self.action_recognizer = recognizer
        self.action_load_error = None
        if self.action_recognizer is None and load_action_model:
            try:
                from logistic_model import LogisticActionRecognizer
                self.action_recognizer = LogisticActionRecognizer()
            except Exception as error:
                self.action_load_error = str(error)
        self.window_seconds = float(window_seconds)
        self.action_interval = float(action_interval)
        self.stale_seconds = float(stale_seconds)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads = []
        self.cap = None
        self.running = False
        self.frame = None
        self.persons = []
        self.active_track_ids = []
        self.frame_count = 0
        self.captured_frame_count = 0
        self.capture_fps = 0.0
        self.last_capture_at = None
        self.inference_fps = 0.0
        self.action_frames = deque(maxlen=512)
        self.action_confidence = None
        self.action_updated_at = None
        self.action_updated_wall_time = None
        self.action_error = None
        self.action_wait_reason = "Collecting a full video window."
        self.processing_error = None
        self.generation = 0
        self.last_action_attempt = float("-inf")
        self.is_violence_detected = False
        self.last_alert_id = 0
        self.alerts = deque(maxlen=100)
        self.source_kind = "none"
        self.source_state = "idle"
        self.source_name = None
        self.source_total_frames = 0
        self.source_fps = 0.0
        self.video_path = None
        self.paused = False
        self.upload_score_complete = False

    def _prepare_source(self, kind, name):
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Previous workers are still stopping. Try again shortly.")
        predictor = getattr(self.detector, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()
        self.stop_event.clear()
        with self.lock:
            self._reset_action_locked()
            self.source_kind, self.source_name = kind, name
            self.source_state = "running" if kind == "camera" else "analysing"
            self.source_total_frames = 0
            self.source_fps = 0.0
            self.upload_score_complete = False
            self.paused = False
            self.processing_error = None
            self.persons = []
            self.active_track_ids = []
            self.frame = None
            self.frame_count = 0
            self.captured_frame_count = 0
            self.capture_fps = 0.0
            self.inference_fps = 0.0
            self.last_capture_at = None
            self.alerts.clear()
            self.running = True

    def start_video(self, path, display_name=None):
        """Score the whole file, then track every frame without skipping frames."""
        if self.running:
            raise RuntimeError("Stop the current source first.")
        path = Path(path)
        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or total < 16 or not math.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError("Choose a readable video with at least 16 frames and valid FPS.")
        try:
            self._prepare_source("upload", display_name or path.name)
        except Exception:
            capture.release()
            raise
        self.cap = capture
        self.video_path = path
        with self.lock:
            self.source_total_frames = total
            self.source_fps = fps
            self.action_wait_reason = "Analysing 16 frames sampled across the uploaded video."
        self.threads = [threading.Thread(target=self._process_upload, daemon=True)]
        self.threads[0].start()

    def set_paused(self, paused):
        with self.lock:
            if self.source_kind != "upload" or self.source_state not in ("running", "paused"):
                raise ValueError("Pause/resume is available during uploaded-video playback.")
            self.paused = bool(paused)
            self.source_state = "paused" if self.paused else "running"

    def _detect_persons(self, frame):
        result = self.detector.track(
            frame, classes=[0], conf=self.confidence_threshold,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )[0]
        track_ids = result.boxes.id
        return [{
            "track_id": int(track_ids[index].item()) if track_ids is not None else None,
            "bbox": [int(value) for value in box.xyxy[0].tolist()],
            "confidence": round(float(box.conf[0]), 4),
        } for index, box in enumerate(result.boxes)]

    def _process_upload(self):
        try:
            if self.action_recognizer is not None:
                try:
                    result = self.action_recognizer.predict_video(self.video_path)
                    now = time.monotonic()
                    self._publish_action(result["fight_score"], self.generation, now, 0, now=now)
                    with self.lock:
                        if not self.stop_event.is_set():
                            self.upload_score_complete = True
                except Exception as error:
                    with self.lock:
                        self.action_error = str(error)
            if self.stop_event.is_set():
                return
            with self.lock:
                self.source_state = "running"
            while not self.stop_event.is_set():
                if self.paused:
                    self.stop_event.wait(0.05)
                    continue
                started = time.monotonic()
                ok, frame = self.cap.read()
                if not ok:
                    with self.lock:
                        if self.frame_count < self.source_total_frames:
                            self.processing_error = "Video ended before all reported frames could be decoded."
                            self.source_state = "error"
                        else:
                            self.source_state = "completed"
                    break
                try:
                    persons = self._detect_persons(frame)
                    detection_error = None
                except Exception as error:
                    persons = []
                    detection_error = f"Person detection failed: {error}"
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.frame = frame
                    self.persons = persons
                    self.active_track_ids = sorted(p["track_id"] for p in persons if p["track_id"] is not None)
                    self.frame_count += 1
                    self.captured_frame_count = self.frame_count
                    self.processing_error = detection_error
                    self.inference_fps = round(1 / max(time.monotonic() - started, 0.001), 1)
                # Never play faster than the source; slow inference slows playback.
                self.stop_event.wait(max(0.0, 1 / self.source_fps - (time.monotonic() - started)))
        except Exception as error:
            with self.lock:
                self.processing_error = str(error)
                self.source_state = "error"
        finally:
            self.cap.release()
            self.running = False

    @staticmethod
    def _threshold(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Threshold must be finite.")
        return max(0.0, min(1.0, value))

    def start(self):
        if self.running:
            return
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("Previous camera workers have not finished stopping.")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            self._prepare_source("camera", f"Camera {self.camera_index}")
        except Exception:
            self.cap.release()
            raise
        self.threads = [threading.Thread(target=self._capture_frames, daemon=True),
                        threading.Thread(target=self._process_frames, daemon=True)]
        if self.action_recognizer is not None:
            self.threads.append(threading.Thread(target=self._classify_clips, daemon=True))
        for thread in self.threads:
            thread.start()

    def _reset_action_locked(self):
        self.generation += 1
        self.action_frames.clear()
        self.action_confidence = None
        self.action_updated_at = None
        self.action_updated_wall_time = None
        self.action_error = None
        self.action_wait_reason = "Collecting a full video window."
        self.last_action_attempt = float("-inf")
        self.is_violence_detected = False

    def _record_action_frame(self, frame, captured_at):
        # Resizing before buffering avoids storing hundreds of full-size images.
        small = cv2.resize(frame, (160, 160))
        with self.lock:
            self.action_frames.append((captured_at, small))
            cutoff = captured_at - self.window_seconds
            while len(self.action_frames) > 1 and self.action_frames[1][0] <= cutoff:
                self.action_frames.popleft()

    def _sample_action_clip(self, now):
        with self.lock:
            if self.action_recognizer is None:
                return None
            if len(self.action_frames) < 16:
                self.action_wait_reason = f"Collecting frames: {len(self.action_frames)}/16."
                return None
            entries = list(self.action_frames)
            if now - entries[-1][0] > self.stale_seconds:
                self.action_wait_reason = "Waiting for fresh camera frames."
                return None
            if entries[-1][0] - entries[0][0] < self.window_seconds:
                self.action_wait_reason = "Collecting a full five-second camera window."
                return None
            if now - self.last_action_attempt < self.action_interval:
                return None
            times = np.array([entry[0] for entry in entries])
            targets = np.linspace(times[-1] - self.window_seconds, times[-1], 16)
            indices = [int(np.abs(times - target).argmin()) for target in targets]
            # Don't pretend repeated frames or a long capture gap are a full clip.
            if len(set(indices)) != 16 or np.diff(times).max() > self.window_seconds / 15:
                self.action_wait_reason = "Camera frames are too sparse or interrupted; collecting a new window."
                return None
            self.last_action_attempt = now
            self.action_wait_reason = "Calculating the Fight score."
            return ([entries[index][1] for index in indices], self.generation,
                    float(times[-1]), self.frame_count)

    def _publish_action(self, score, generation, clip_end, frame_number, now=None):
        now = time.monotonic() if now is None else now
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid action score.")
        with self.lock:
            if generation != self.generation or self.stop_event.is_set():
                return
            if now - clip_end > self.stale_seconds:
                return
            was_detected = self.is_violence_detected
            if self.action_updated_at is None or now - self.action_updated_at > self.stale_seconds:
                was_detected = False
            self.action_confidence = score
            self.action_updated_at = clip_end
            self.action_updated_wall_time = time.time()
            self.action_error = None
            self.action_wait_reason = None
            self._apply_alert_policy_locked(score, frame_number, was_detected)

    def _apply_alert_policy_locked(self, score, frame_number, was_detected):
        self.is_violence_detected = self.alerts_enabled and score >= self.alert_threshold
        if self.is_violence_detected and not was_detected:
            self.last_alert_id += 1
            self.alerts.append({
                "id": self.last_alert_id, "timestamp": time.time(),
                "confidence": round(score, 4), "frame_number": frame_number,
                "status": "ACTIVE", "type": "POSSIBLE_VIOLENCE",
                "model": "MobileNetV3 + logistic regression", "demo": True,
                "threshold": self.alert_threshold,
                "score_scope": "whole_video" if self.source_kind == "upload" else "rolling_window",
            })

    def set_alert_threshold(self, threshold):
        """Enable demo alerts and immediately recheck valid existing evidence."""
        value = float(threshold)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Alert threshold must be between 0 and 1.")
        now = time.monotonic()
        with self.lock:
            self.alert_threshold = value
            self.alerts_enabled = True
            fresh = (self.action_confidence is not None
                     and self.action_updated_at is not None
                     and self.action_error is None
                     and self.source_state not in ("idle", "stopped", "error")
                     and ((self.source_kind == "upload" and self.upload_score_complete)
                          or now - self.action_updated_at <= self.stale_seconds))
            if fresh:
                self._apply_alert_policy_locked(
                    self.action_confidence, self.frame_count, self.is_violence_detected
                )
            else:
                self.is_violence_detected = False

    def _classify_clips(self):
        while not self.stop_event.wait(0.05):
            sampled = self._sample_action_clip(time.monotonic())
            if sampled is None:
                continue
            frames, generation, clip_end, frame_number = sampled
            try:
                self._publish_action(self.action_recognizer.predict(frames),
                                     generation, clip_end, frame_number)
            except Exception as error:
                with self.lock:
                    if generation == self.generation:
                        self.action_error = str(error)
                        self.action_confidence = None
                        self.action_updated_at = None
                        self.is_violence_detected = False

    def _capture_frames(self):
        """Read the camera independently of potentially slow YOLO inference."""
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                with self.lock:
                    self._reset_action_locked()
                    self.frame = None
                    self.persons = []
                    self.active_track_ids = []
                    self.processing_error = "Camera did not return a frame."
                    self.last_capture_at = None
                    self.capture_fps = 0.0
                self.stop_event.wait(0.1)
                continue
            captured_at = time.monotonic()
            with self.lock:
                if self.last_capture_at is not None:
                    self.capture_fps = round(1 / max(captured_at - self.last_capture_at, 0.001), 1)
                self.last_capture_at = captured_at
                self.frame = frame.copy()
                self.captured_frame_count += 1
                if self.processing_error == "Camera did not return a frame.":
                    self.processing_error = None
            if self.action_recognizer is not None:
                self._record_action_frame(frame, captured_at)

    def _process_frames(self):
        """Track the newest available frame, skipping backlog instead of capture."""
        last_processed = -1
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
                sequence = self.captured_frame_count
                generation = self.generation
            if frame is None or sequence == last_processed:
                self.stop_event.wait(0.01)
                continue
            last_processed = sequence
            started = time.monotonic()
            try:
                persons = self._detect_persons(frame)
                with self.lock:
                    self.processing_error = None
            except Exception as error:
                persons = []
                with self.lock:
                    self.processing_error = f"Person detection failed: {error}"
            with self.lock:
                # A camera failure/reset invalidates detections computed earlier.
                if generation != self.generation:
                    continue
                self.persons = persons
                self.active_track_ids = sorted(p["track_id"] for p in persons if p["track_id"] is not None)
                self.frame_count += 1
                self.inference_fps = round(1 / max(time.monotonic() - started, 0.001), 1)

    def get_data(self):
        now = time.monotonic()
        with self.lock:
            persons = [dict(person) for person in self.persons]
            age = None if self.action_updated_at is None else max(0.0, now - self.action_updated_at)
            fresh = (age is not None and self.action_error is None and
                     ((self.source_kind == "upload" and self.upload_score_complete)
                      or age <= self.stale_seconds))
            score = self.action_confidence if fresh else None
            if self.action_recognizer is None:
                status = "unavailable"
            elif self.source_state in ("idle", "stopped"):
                status = "idle"
                fresh = False
                score = None
            elif self.action_error:
                status = "error"
            elif self.action_updated_at is None:
                status = "warming_up"
            else:
                status = "ready" if fresh else "stale"
            return {
                "human_detected": bool(persons), "person_count": len(persons),
                "source_kind": self.source_kind, "source_state": self.source_state,
                "source_revision": self.generation,
                "source_name": self.source_name,
                "source_total_frames": self.source_total_frames,
                "source_frame": self.frame_count if self.source_kind == "upload" else None,
                "source_duration_seconds": self.source_total_frames / self.source_fps if self.source_fps else None,
                "source_position_seconds": self.frame_count / self.source_fps if self.source_fps else None,
                "source_progress": round(min(1.0, self.frame_count / self.source_total_frames), 4) if self.source_total_frames else None,
                "score_scope": "whole_video" if self.source_kind == "upload" else "rolling_window",
                "persons": persons, "active_track_ids": list(self.active_track_ids),
                "confidence_threshold": self.confidence_threshold,
                "frame_count": self.frame_count, "inference_fps": self.inference_fps,
                "captured_frame_count": self.captured_frame_count,
                "capture_fps": self.capture_fps,
                "model": "YOLO11n (COCO pretrained)",
                "action_model_ready": self.action_recognizer is not None,
                "action_model": "MobileNetV3 + logistic regression" if self.action_recognizer is not None else None,
                "action_status": status, "action_error": self.action_load_error or self.action_error,
                "action_wait_reason": self.action_wait_reason,
                "action_buffer_frames": len(self.action_frames),
                "action_buffer_seconds": round(self.action_frames[-1][0] - self.action_frames[0][0], 2) if len(self.action_frames) > 1 else 0.0,
                "processing_error": self.processing_error,
                "violence_confidence": None if score is None else round(score, 6),
                "classification_threshold": 0.5, "violence_threshold": self.alert_threshold,
                "alerts_enabled": self.alerts_enabled,
                "alert_policy": "Unvalidated demo policy; human review required",
                "action_window_seconds": self.window_seconds,
                "action_score_age_seconds": None if age is None else round(age, 2),
                "action_updated_at": self.action_updated_wall_time,
                "is_violence_detected": bool(fresh and self.is_violence_detected),
                "confidence": max((p["confidence"] for p in persons), default=0.0),
                "alerts_count": len(self.alerts),
            }

    def get_frame_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
            persons = [dict(person) for person in self.persons]
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            track_id = person["track_id"] if person["track_id"] is not None else "?"
            cv2.putText(frame, f"Person #{track_id} {person['confidence']:.0%}",
                        (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        data = self.get_data()
        cv2.putText(frame, f"People detected: {len(persons)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        score = data["violence_confidence"]
        label = f"Fight score: {score:.0%}" if score is not None else f"Action: {data['action_status']}"
        cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def get_alerts(self, limit=10):
        with self.lock:
            return [dict(alert) for alert in reversed(self.alerts)][:max(0, min(100, limit))]

    def set_threshold(self, threshold):
        value = self._threshold(threshold)
        with self.lock:
            self.confidence_threshold = value

    def acknowledge_alert(self, alert_id):
        with self.lock:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["status"] = "ACKNOWLEDGED"
                    return True
        return False

    def stop(self):
        self.stop_event.set()
        self.running = False
        for thread in self.threads:
            thread.join(timeout=3)
        if self.cap is not None:
            self.cap.release()
        with self.lock:
            self._reset_action_locked()
            self.frame = None
            self.persons = []
            self.active_track_ids = []
            self.source_state = "stopped" if self.source_kind != "none" else "idle"
