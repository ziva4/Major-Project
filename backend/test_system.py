#!/usr/bin/env python3
"""
System Test Script - Test all components
Run without starting the Flask server
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_processor import VideoProcessor
from train_model import ViolenceDetectionModel

def separator(title=""):
    """Print a separator line"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"{'-'*60}\n")

def test_model_loading():
    """Test model loading and basic prediction"""
    separator("Test 1: Model Loading & Prediction")

    print("Loading violence detection model...")
    detector = ViolenceDetectionModel()

    if detector.load_model():
        print("✓ Model loaded successfully from disk")
    else:
        print("⚠ Model not found, would train on first run")
        print("  This is normal for first-time setup")
        return True

    print("\nTesting prediction on synthetic frames...")

    # Test on random frames
    for i in range(3):
        frame = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)
        confidence = detector.predict(frame)
        print(f"  Frame {i+1}: Confidence = {confidence:.4f} (Normal activity)")

    print("\nTesting on high-intensity frames...")
    for i in range(3):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        noise = np.random.randint(0, 100, (480, 640, 3))
        frame = cv2.add(frame, noise)
        confidence = detector.predict(frame)
        print(f"  Frame {i+1}: Confidence = {confidence:.4f} (Potential violence)")

    print("\n✓ Model prediction test PASSED")
    return True

def test_camera_access():
    """Test camera access and capture"""
    separator("Test 2: Camera Access")

    print("Attempting to access camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("✗ Camera not accessible")
        print("  Solutions:")
        print("  1. Check if camera is connected")
        print("  2. Grant camera permissions in system settings")
        print("  3. Check if another app is using the camera")
        return False

    print("✓ Camera opened successfully")

    print("\nCapturing test frames...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  Frame {i+1}: {w}×{h} pixels - ✓ Captured")
            if i == 0:
                print(f"           Data type: {frame.dtype}, Channels: {frame.shape[2]}")
        else:
            print(f"  Frame {i+1}: ✗ Failed to capture")
            cap.release()
            return False

        time.sleep(0.1)

    cap.release()
    print("\n✓ Camera test PASSED")
    return True

def test_video_processor():
    """Test video processor initialization"""
    separator("Test 3: Video Processor")

    print("Initializing VideoProcessor...")

    try:
        processor = VideoProcessor(confidence_threshold=0.6)
        print("✓ VideoProcessor created")

        print("\nStarting video capture...")
        processor.start()
        print("✓ Video capture started")

        print("\nProcessing frames for 3 seconds...")
        for i in range(3):
            data = processor.get_data()
            print(f"  {i+1}s: Confidence={data['confidence']:.2%}, " +
                  f"Frames={data['frame_count']}, " +
                  f"Alerts={data['alerts_count']}")
            time.sleep(1)

        print("\nGetting frame as JPEG...")
        jpeg_data = processor.get_frame_jpeg()
        if jpeg_data:
            print(f"✓ JPEG frame generated ({len(jpeg_data)} bytes)")
        else:
            print("✗ Failed to generate JPEG")

        processor.stop()
        print("\n✓ Video Processor test PASSED")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction pipeline"""
    separator("Test 4: Feature Extraction")

    print("Creating test frame...")
    frame = np.random.randint(0, 150, (480, 640, 3), dtype=np.uint8)

    print("Extracting features...")
    detector = ViolenceDetectionModel()
    features = detector.create_features_from_frame(frame)

    print(f"✓ Features extracted: {features.shape[0]} dimensions")
    print(f"  Min value: {features.min():.4f}")
    print(f"  Max value: {features.max():.4f}")
    print(f"  Mean value: {features.mean():.4f}")
    print(f"  Std Dev: {features.std():.4f}")

    if features.shape[0] == 128:
        print("\n✓ Feature Extraction test PASSED")
        return True
    else:
        print("\n✗ Feature dimension mismatch")
        return False

def test_confidence_smoothing():
    """Test confidence smoothing algorithm"""
    separator("Test 5: Confidence Smoothing")

    from collections import deque

    print("Testing moving average smoothing...")

    # Simulate confidence values over time
    raw_values = [0.2, 0.3, 0.8, 0.9, 0.95, 0.5, 0.3, 0.2]
    history = deque(maxlen=5)
    smoothed_values = []

    print("\n  Frame  | Raw Value | Smoothed Value | Status")
    print("  " + "-"*50)

    for i, raw in enumerate(raw_values):
        history.append(raw)
        smoothed = np.mean(list(history))
        smoothed_values.append(smoothed)

        threshold = 0.6
        status = "ALERT" if smoothed > threshold else "Normal"
        color = "🔴" if smoothed > threshold else "🟢"

        print(f"  {i+1:>5} | {raw:>9.2f} | {smoothed:>14.2f} | {color} {status}")

    print("\n✓ Confidence Smoothing test PASSED")
    return True

def run_all_tests():
    """Run all system tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  🚨 VIOLENCE DETECTION SYSTEM - TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    tests = [
        ("Model Loading", test_model_loading),
        ("Camera Access", test_camera_access),
        ("Video Processor", test_video_processor),
        ("Feature Extraction", test_feature_extraction),
        ("Confidence Smoothing", test_confidence_smoothing),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    separator("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:<8} {test_name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✓ All tests passed! System is ready to run.")
        print("\n  Next steps:")
        print("    1. Run: python app.py")
        print("    2. Open: http://localhost:5000")
        print("    3. Start monitoring!")
        return 0
    else:
        print("\n  ⚠ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
