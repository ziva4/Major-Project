"""Run with the downloaded bundle: python -m unittest discover -s backend -p test_logistic_model.py."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from logistic_model import LogisticActionRecognizer


class LogisticModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.recognizer = LogisticActionRecognizer()

    def test_preprocessing_and_pooling_match_colab_reference(self):
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(16)]
        # Independently reproduce the notebook's validation preprocessing.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        images = []
        for frame in frames:
            image = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = np.clip((image - 0.5) * 1.0 + 0.5 + 0.0, 0.0, 1.0)
            images.append(np.transpose((image - mean) / std, (2, 0, 1)))
        tensor = torch.from_numpy(np.stack(images)).float()
        with torch.inference_mode():
            features = self.recognizer.cnn(tensor).mean(dim=(-2, -1)).reshape(1, 16, 576)
            reference = torch.cat((features.mean(1), features.std(1, unbiased=False)), 1).numpy()
        actual = self.recognizer.extract_features(frames)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
        expected_score = self.recognizer.classifier.predict_proba(reference)[0, 1]
        self.assertAlmostEqual(self.recognizer.predict(frames), expected_score, places=5)

    def test_video_sampling_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(32):
                    writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
            finally:
                writer.release()
            result = self.recognizer.predict_video(path)
            self.assertEqual(result["sampled_frame_indices"], np.linspace(0, 31, 16).astype(int).tolist())
            self.assertEqual(result["total_frames"], 32)
            self.assertTrue(0.0 <= result["fight_score"] <= 1.0)
            self.assertEqual(result["predicted_class"], "Fight" if result["fight_score"] >= 0.5 else "NonFight")

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.recognizer.predict([np.zeros((10, 10, 3), dtype=np.uint8)] * 15)
        with self.assertRaises(ValueError):
            self.recognizer.predict([np.zeros((10, 10), dtype=np.uint8)] * 16)
        with self.assertRaises(FileNotFoundError):
            self.recognizer.predict_video(Path("nonexistent-video-for-test.avi"))


if __name__ == "__main__":
    unittest.main()
