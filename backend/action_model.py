"""CNN/LSTM model and runtime wrapper for short violence-action clips."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ViolenceCNNLSTM(nn.Module):
    """MobileNetV3 extracts each frame; an LSTM learns movement over time."""

    def __init__(self, hidden_size=256, lstm_layers=1, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(576, hidden_size, lstm_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(hidden_size, 1)
        )

    def forward(self, clips):
        batch, frames, channels, height, width = clips.shape
        features = self.cnn(clips.reshape(batch * frames, channels, height, width))
        features = self.pool(features).flatten(1).reshape(batch, frames, -1)
        sequence, _ = self.lstm(features)
        return self.classifier(sequence[:, -1]).squeeze(1)


class ViolenceActionRecognizer:
    """Loads an optional trained checkpoint and scores BGR OpenCV frames."""

    def __init__(self, model_path=None):
        self.model_path = Path(model_path or Path(__file__).parent / "models" / "violence_action_model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sequence_length = 32
        self.image_size = 160
        self.threshold = 0.80
        self.load_error = None
        self._load()

    @property
    def ready(self):
        return self.model is not None

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            self.sequence_length = int(config.get("sequence_length", self.sequence_length))
            self.image_size = int(config.get("image_size", self.image_size))
            self.threshold = float(config.get("threshold", self.threshold))
            self.model = ViolenceCNNLSTM(
                hidden_size=int(config.get("hidden_size", 256)),
                lstm_layers=int(config.get("lstm_layers", 1)),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        except Exception as error:
            self.model = None
            self.load_error = str(error)

    def predict(self, frames):
        if not self.ready or len(frames) != self.sequence_length:
            return 0.0
        prepared = []
        for frame in frames:
            image = cv2.resize(frame, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            prepared.append(np.transpose(image, (2, 0, 1)))
        clip = torch.from_numpy(np.stack(prepared)).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return float(torch.sigmoid(self.model(clip))[0].item())
