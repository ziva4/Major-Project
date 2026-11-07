"""Inference for the exported MobileNetV3 mean/std + logistic classifier."""

import json
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
import torch
from sklearn.exceptions import InconsistentVersionWarning
from torchvision.models import mobilenet_v3_small


DEFAULT_BUNDLE = Path(__file__).parent / "models" / "mobilenet_logistic"


class LogisticActionRecognizer:
    """Load a trusted local bundle and score 16 BGR frames or a video file.

    This class does not apply live alert thresholds or score smoothing.
    """

    def __init__(self, bundle_dir=DEFAULT_BUNDLE, device="cpu"):
        self.bundle_dir = Path(bundle_dir)
        required = (
            "feature_metadata.json", "package_versions.json",
            "best_logistic_classifier.joblib", "mobilenet_feature_extractor.pt",
        )
        for name in required:
            if not (self.bundle_dir / name).is_file():
                raise FileNotFoundError(f"Missing model artifact: {self.bundle_dir / name}")

        self.metadata = json.loads(
            (self.bundle_dir / "feature_metadata.json").read_text(encoding="utf-8")
        )
        expected = {
            "method": "mobilenet_v3_small_imagenet_mean_std_v1",
            "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            "frames": 16, "image_size": 160,
            "sampling": "uniform_endpoints", "normalization": "imagenet",
            "augmentation": False,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"Unsupported feature setting {key}: {self.metadata.get(key)!r}")

        self.versions = json.loads(
            (self.bundle_dir / "package_versions.json").read_text(encoding="utf-8")
        )
        if self.versions.get("sklearn") != sklearn.__version__:
            raise RuntimeError(
                f"Classifier requires scikit-learn {self.versions.get('sklearn')}; "
                f"installed: {sklearn.__version__}. Use the matching version."
            )

        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Choose cpu or cuda for inference.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        # weights=None prevents network downloads. Load the exported CNN exactly.
        self.cnn = mobilenet_v3_small(weights=None).features
        state = torch.load(
            self.bundle_dir / "mobilenet_feature_extractor.pt",
            map_location="cpu", weights_only=True,
        )
        self.cnn.load_state_dict(state, strict=True)
        self.cnn.to(self.device).eval()
        self.cnn.requires_grad_(False)

        # joblib files must come from the user's trusted experiment bundle.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            self.classifier = joblib.load(
                self.bundle_dir / "best_logistic_classifier.joblib"
            )
        if list(self.classifier.named_steps) != ["scaler", "classifier"]:
            raise ValueError("Expected the exported scaler/classifier pipeline.")
        if not np.array_equal(self.classifier.classes_, [0, 1]):
            raise ValueError("Expected class labels 0=NonFight, 1=Fight.")
        if self.classifier.n_features_in_ != 1152:
            raise ValueError("Expected 1,152 pooled features.")
        if self.classifier.named_steps["classifier"].C != 0.01:
            raise ValueError("Expected the selected logistic classifier with C=0.01.")

        self.sequence_length = 16
        self.image_size = 160
        self.classification_threshold = 0.5

    def extract_features(self, frames):
        """Match the deterministic Colab preprocessing and population std."""
        if len(frames) != self.sequence_length:
            raise ValueError(f"Expected exactly {self.sequence_length} frames.")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        prepared = []
        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
                raise ValueError("Frames must be nonempty uint8 BGR images with 3 channels.")
            image = cv2.resize(frame, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Validation preprocessing in Colab also clipped to [0, 1].
            image = np.clip(image, 0.0, 1.0)
            image = (image - mean) / std
            prepared.append(np.transpose(image, (2, 0, 1)))

        tensor = torch.from_numpy(np.stack(prepared)).float().to(self.device)
        with torch.inference_mode():
            features = self.cnn(tensor).mean(dim=(-2, -1)).unsqueeze(0)
            summary = torch.cat((
                features.mean(dim=1), features.std(dim=1, unbiased=False)
            ), dim=1)
        result = summary.cpu().numpy()
        if result.shape != (1, 1152) or not np.isfinite(result).all():
            raise RuntimeError("Feature extraction produced invalid values.")
        return result

    def predict(self, frames):
        """Return the Fight class score before any alert-policy processing."""
        features = self.extract_features(frames)
        score = float(self.classifier.predict_proba(features)[0, 1])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("Classifier produced an invalid score.")
        return score

    def predict_video(self, video_path):
        """Sample the same uniform frame indices as the Colab evaluation."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < self.sequence_length:
                raise ValueError("Video must report at least 16 frames, matching evaluation.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            indices = np.linspace(0, count - 1, self.sequence_length).astype(int)
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"Cannot decode sampled frame {index}: {video_path}")
                frames.append(frame)
        finally:
            capture.release()

        score = self.predict(frames)
        return {
            "video": str(video_path.resolve()),
            "model": "MobileNetV3-small mean/std + logistic regression",
            "fight_score": score,
            "classification_threshold": self.classification_threshold,
            "predicted_class": "Fight" if score >= self.classification_threshold else "NonFight",
            "sampled_frame_indices": indices.tolist(),
            "total_frames": count,
            "fps": fps if np.isfinite(fps) and fps > 0 else None,
            "device": str(self.device),
            "note": "Offline clip prediction; not confirmation of violence or a live alert.",
        }
