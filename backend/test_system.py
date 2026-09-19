"""Basic checks for the pretrained person-detection pipeline."""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from video_processor import VideoProcessor


def test_camera_access():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL: Camera is not available")
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("FAIL: Camera opened but did not return a frame")
        return False
    print(f"PASS: Camera frame captured ({frame.shape[1]}x{frame.shape[0]})")
    return True


def test_person_detector():
    try:
        processor = VideoProcessor(confidence_threshold=0.50)
        initial = processor.get_data()
        assert initial["human_detected"] is False
        assert initial["person_count"] == 0
        assert initial["model"] == "YOLO11n (COCO pretrained)"
        assert isinstance(initial["action_model_ready"], bool)
        assert initial["violence_confidence"] == 0.0
        print("PASS: YOLO person detector loaded")
        return True
    except Exception as error:
        print(f"FAIL: Could not load person detector: {error}")
        return False


def test_live_processing():
    try:
        processor = VideoProcessor(confidence_threshold=0.50)
        processor.start()
        time.sleep(3)
        data = processor.get_data()
        image = processor.get_frame_jpeg()
        processor.stop()
        assert "persons" in data and "person_count" in data
        assert "active_track_ids" in data
        assert "violence_confidence" in data and "action_model_ready" in data
        assert image
        print(
            "PASS: Live inference works "
            f"({data['person_count']} people, {data['inference_fps']} inference FPS)"
        )
        return True
    except Exception as error:
        print(f"FAIL: Live person detection failed: {error}")
        return False


if __name__ == "__main__":
    results = [test_camera_access(), test_person_detector(), test_live_processing()]
    sys.exit(0 if all(results) else 1)
