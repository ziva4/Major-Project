"""Camera-free tests for timed sampling, model state, alerts, and HTTP/UI wiring."""

import threading
import time
import unittest
import tempfile
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

from video_processor import VideoProcessor


class DummyRecognizer:
    def predict(self, frames):
        return 0.9

    def predict_video(self, path):
        return {"fight_score": 0.9}


class LiveProcessorTests(unittest.TestCase):
    def make_processor(self, **kwargs):
        processor = VideoProcessor(detector=object(), recognizer=DummyRecognizer(), **kwargs)
        processor.source_kind = "camera"
        processor.source_state = "running"
        return processor

    def test_timed_sampling_requires_full_window_and_sixteen_distinct_frames(self):
        processor = self.make_processor()
        for i in range(51):
            frame = np.full((20, 20, 3), i, dtype=np.uint8)
            processor._record_action_frame(frame, 100 + i / 10)
            if i < 50:
                self.assertIsNone(processor._sample_action_clip(100 + i / 10))
        sampled = processor._sample_action_clip(105)
        self.assertIsNotNone(sampled)
        frames, _, end, _ = sampled
        self.assertEqual(len(frames), 16)
        self.assertEqual(len({int(frame[0, 0, 0]) for frame in frames}), 16)
        self.assertEqual(int(frames[0][0, 0, 0]), 0)
        self.assertEqual(int(frames[-1][0, 0, 0]), 50)
        self.assertEqual(end, 105)
        self.assertIsNone(processor._sample_action_clip(105.5))

    def test_long_camera_gap_does_not_produce_a_score(self):
        processor = self.make_processor()
        for timestamp in list(np.linspace(100, 102, 25)) + list(np.linspace(104, 105, 15)):
            processor._record_action_frame(np.zeros((20, 20, 3), dtype=np.uint8), timestamp)
        self.assertIsNone(processor._sample_action_clip(105))

    def test_loaded_model_has_no_score_during_warmup(self):
        data = self.make_processor().get_data()
        self.assertTrue(data["action_model_ready"])
        self.assertEqual(data["action_status"], "warming_up")
        self.assertIsNone(data["violence_confidence"])
        self.assertFalse(data["is_violence_detected"])

    def test_stale_score_is_not_displayed_or_alerted(self):
        processor = self.make_processor(alerts_enabled=True)
        processor._publish_action(0.9, 0, 100, 1, now=100)
        with patch("video_processor.time.monotonic", return_value=104):
            data = processor.get_data()
        self.assertEqual(data["action_status"], "stale")
        self.assertIsNone(data["violence_confidence"])
        self.assertFalse(data["is_violence_detected"])

    def test_scores_do_not_generate_alerts_by_default(self):
        processor = self.make_processor()
        processor._publish_action(0.99, 0, 100, 1, now=100)
        self.assertEqual(processor.get_alerts(), [])

    def test_demo_alerts_trigger_on_transitions_and_can_be_acknowledged(self):
        processor = self.make_processor(alerts_enabled=True)
        for score, timestamp in [(0.739, 100), (0.75, 101), (0.69, 102), (0.70, 103)]:
            processor._publish_action(score, 0, timestamp, 1, now=timestamp)
        self.assertEqual(len(processor.get_alerts()), 2)
        self.assertTrue(processor.acknowledge_alert(1))
        self.assertFalse(processor.acknowledge_alert(100))
        self.assertEqual(processor.get_alerts()[-1]["status"], "ACKNOWLEDGED")

    def test_camera_reset_discards_inflight_prediction(self):
        processor = self.make_processor()
        with processor.lock:
            processor._reset_action_locked()
        processor._publish_action(0.9, 0, 100, 1, now=100)
        self.assertIsNone(processor.action_confidence)

    def test_slider_enables_alerts_rechecks_score_and_avoids_duplicates(self):
        processor = self.make_processor()
        processor._publish_action(0.739, 0, 100, 1, now=100)
        with patch("video_processor.time.monotonic", return_value=101):
            processor.set_alert_threshold(0.80)
            self.assertTrue(processor.alerts_enabled)
            self.assertEqual(processor.get_alerts(), [])
            processor.set_alert_threshold(0.70)
            self.assertEqual(len(processor.get_alerts()), 1)
            processor.set_alert_threshold(0.65)
            processor._publish_action(0.74, 0, 101, 2, now=101)
            self.assertEqual(len(processor.get_alerts()), 1)
            processor.set_alert_threshold(0.75)
            self.assertFalse(processor.is_violence_detected)
            processor.set_alert_threshold(0.70)
            self.assertEqual(len(processor.get_alerts()), 2)
        self.assertEqual(processor.confidence_threshold, 0.5)
        self.assertEqual(processor.get_alerts()[0]["threshold"], 0.70)

    def test_slider_does_not_alert_on_stale_live_scores_or_without_evidence(self):
        processor = self.make_processor()
        processor.set_alert_threshold(0)
        self.assertEqual(processor.get_alerts(), [])
        processor._publish_action(0.739, 0, 100, 1, now=100)
        count = len(processor.get_alerts())
        with patch("video_processor.time.monotonic", return_value=105):
            processor.set_alert_threshold(0.70)
        self.assertEqual(len(processor.get_alerts()), count)
        self.assertFalse(processor.is_violence_detected)

    def test_slider_rechecks_completed_upload_even_after_playback_ends(self):
        processor = self.make_processor()
        processor.source_kind = "upload"
        processor.source_state = "completed"
        processor.upload_score_complete = True
        processor._publish_action(0.739, 0, 100, 1, now=100)
        with patch("video_processor.time.monotonic", return_value=200):
            processor.set_alert_threshold(0.70)
        self.assertEqual(len(processor.get_alerts()), 1)
        self.assertEqual(processor.get_alerts()[0]["score_scope"], "whole_video")

    def test_slider_threshold_includes_exact_boundary(self):
        processor = self.make_processor()
        processor.set_alert_threshold(0.70)
        processor._publish_action(0.6999, 0, 100, 1, now=100)
        self.assertEqual(len(processor.get_alerts()), 0)
        processor._publish_action(0.70, 0, 101, 2, now=101)
        self.assertEqual(len(processor.get_alerts()), 1)

    def test_model_load_failure_preserves_person_api(self):
        with patch("logistic_model.LogisticActionRecognizer", side_effect=RuntimeError("missing bundle")):
            processor = VideoProcessor(detector=object())
        data = processor.get_data()
        self.assertEqual(data["action_status"], "unavailable")
        self.assertEqual(data["action_error"], "missing bundle")
        self.assertEqual(data["person_count"], 0)

    def test_slow_classifier_does_not_hold_api_lock(self):
        processor = self.make_processor()
        started = threading.Event()
        release = threading.Event()

        def slow_predict(frames):
            started.set()
            release.wait(3)
            return 0.9

        processor.action_recognizer.predict = slow_predict
        now = time.monotonic()
        for i in range(51):
            processor._record_action_frame(np.zeros((20, 20, 3), dtype=np.uint8), now - 5 + i / 10)
        worker = threading.Thread(target=processor._classify_clips)
        worker.start()
        try:
            self.assertTrue(started.wait(3))
            with processor.lock:
                processor.frame_count = 42
            self.assertEqual(processor.get_data()["frame_count"], 42)
        finally:
            processor.stop_event.set()
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())

    def test_camera_and_action_buffer_keep_advancing_while_detector_is_blocked(self):
        entered = threading.Event()
        release = threading.Event()

        class FakeCamera:
            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                time.sleep(0.01)
                return True, np.zeros((20, 20, 3), dtype=np.uint8)

            def release(self):
                pass

        class BlockedDetector:
            def track(self, *args, **kwargs):
                entered.set()
                release.wait(5)
                # An iterable empty collection with an .id tracking attribute.
                class EmptyBoxes:
                    id = None
                    def __iter__(self):
                        return iter(())
                return [SimpleNamespace(boxes=EmptyBoxes())]

        processor = VideoProcessor(detector=BlockedDetector(), recognizer=DummyRecognizer())
        with patch("video_processor.cv2.VideoCapture", return_value=FakeCamera()):
            processor.start()
            try:
                self.assertTrue(entered.wait(3))
                deadline = time.monotonic() + 3
                while processor.get_data()["action_buffer_frames"] < 20 and time.monotonic() < deadline:
                    time.sleep(0.02)
                data = processor.get_data()
                self.assertGreaterEqual(data["action_buffer_frames"], 20)
                self.assertGreaterEqual(data["captured_frame_count"], 20)
                self.assertEqual(data["frame_count"], 0)
                self.assertIsNotNone(processor.get_frame_jpeg())
            finally:
                release.set()
                processor.stop()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        import app
        self.module = app
        self.previous = app.processor
        app.processor = VideoProcessor(detector=object(), recognizer=DummyRecognizer())
        self.client = app.app.test_client()

    def tearDown(self):
        self.module.processor.stop()
        self.module.processor = self.previous

    def test_dashboard_and_detection_contract(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="fight-score"', page.data)
        data = self.client.get("/api/detection").get_json()
        self.assertEqual(data["action_model"], "MobileNetV3 + logistic regression")
        self.assertIsNone(data["violence_confidence"])

    def test_person_threshold_is_separate_and_invalid_values_are_rejected(self):
        response = self.client.post("/api/threshold", json={"threshold": 0.7})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.module.processor.alert_threshold, 0.7)
        for value in (None, True, "nan", {}, []):
            self.assertEqual(self.client.post("/api/threshold", json={"threshold": value}).status_code, 400)
        self.assertEqual(self.client.post("/api/alerts/999/acknowledge").status_code, 404)

    def test_demo_threshold_api_enables_alerts_without_changing_person_confidence(self):
        response = self.client.post("/api/alert-threshold", json={"threshold": 0.6})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["threshold"], 0.6)
        self.assertTrue(response.get_json()["alerts_enabled"])
        self.assertEqual(self.module.processor.confidence_threshold, 0.5)
        for value in (None, True, "nan", "inf", -0.1, 1.1, {}, []):
            self.assertEqual(self.client.post("/api/alert-threshold", json={"threshold": value}).status_code, 400)
        self.assertEqual(self.module.processor.alert_threshold, 0.6)

    def test_socket_update_uses_current_model_state(self):
        socket_client = self.module.socketio.test_client(self.module.app)
        try:
            socket_client.get_received()
            socket_client.emit("request_update")
            received = socket_client.get_received()
            event = next(item for item in received if item["name"] == "detection_update")
            self.assertEqual(event["args"][0]["action_status"], "idle")
        finally:
            socket_client.disconnect()

    def test_idle_dashboard_does_not_open_camera(self):
        with patch("video_processor.cv2.VideoCapture") as camera:
            self.client.get("/")
            data = self.client.get("/api/detection").get_json()
        camera.assert_not_called()
        self.assertEqual(data["source_state"], "idle")
        self.assertEqual(data["action_status"], "idle")

    def test_upload_validation_rejects_missing_wrong_type_and_bad_content(self):
        self.assertEqual(self.client.post("/api/source/upload").status_code, 400)
        response = self.client.post("/api/source/upload", data={"video": (io.BytesIO(b"bad"), "file.txt")})
        self.assertEqual(response.status_code, 415)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(self.module.app.config, UPLOAD_FOLDER=directory):
                response = self.client.post("/api/source/upload", data={"video": (io.BytesIO(b"bad"), "file.avi")})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(self.module.processor.source_state, "idle")

    def test_upload_limit_returns_json(self):
        with patch.dict(self.module.app.config, MAX_CONTENT_LENGTH=32):
            response = self.client.post("/api/source/upload", data={"video": (io.BytesIO(b"x" * 100), "file.avi")})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_source_changes_are_serialized(self):
        self.module.source_lock.acquire()
        try:
            self.assertEqual(self.client.post("/api/source/live").status_code, 409)
            self.assertEqual(self.client.post("/api/source/stop").status_code, 409)
        finally:
            self.module.source_lock.release()

    def test_uploaded_video_processes_all_frames_and_keeps_final_score(self):
        class EmptyBoxes:
            id = None
            def __iter__(self):
                return iter(())

        self.module.processor.detector = SimpleNamespace(
            track=lambda *a, **kw: [SimpleNamespace(boxes=EmptyBoxes())]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 100, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(32):
                writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
            writer.release()
            upload_folder = Path(directory) / "uploads"
            with patch.dict(self.module.app.config, UPLOAD_FOLDER=upload_folder):
                response = self.client.post("/api/source/upload", data={
                    "video": (io.BytesIO(input_path.read_bytes()), "../../sample.avi")
                })
            self.assertEqual(response.status_code, 200, response.get_json())
            deadline = time.monotonic() + 5
            while self.module.processor.running and time.monotonic() < deadline:
                time.sleep(0.02)
            data = self.module.processor.get_data()
            self.assertEqual(data["source_state"], "completed")
            self.assertEqual(data["source_name"], "sample.avi")
            self.assertEqual(data["source_frame"], 32)
            self.assertEqual(data["source_progress"], 1.0)
            self.assertEqual(data["score_scope"], "whole_video")
            with patch("video_processor.time.monotonic", return_value=time.monotonic() + 100):
                self.assertEqual(self.module.processor.get_data()["violence_confidence"], 0.9)
            self.assertEqual(len(list(upload_folder.iterdir())), 1)
            self.assertEqual(self.client.post("/api/source/stop").status_code, 200)
            self.assertIsNone(self.module.processor.get_data()["violence_confidence"])

    def test_pause_resume_validation(self):
        self.assertEqual(self.client.post("/api/source/playback", json={"action": "pause"}).status_code, 409)
        self.module.processor.source_kind = "upload"
        self.module.processor.source_state = "running"
        self.assertEqual(self.client.post("/api/source/playback", json={"action": "pause"}).status_code, 200)
        self.assertTrue(self.module.processor.paused)
        self.assertEqual(self.client.post("/api/source/playback", json={"action": "resume"}).status_code, 200)
        self.assertFalse(self.module.processor.paused)
        self.assertEqual(self.client.post("/api/source/playback", json={"action": "seek"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
