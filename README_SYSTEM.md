# Violence Detection and Person-Tracking System

This is an educational, local surveillance-dashboard prototype. It combines
person detection and tracking with an optional video-action classifier that
flags *possible violence* for human review. It is not suitable for autonomous
safety, law-enforcement, or disciplinary decisions.

## Current system

```text
Webcam
  -> OpenCV capture
  -> YOLO11n person detection + ByteTrack IDs
  -> MobileNetV3 + LSTM action score (optional)
  -> Flask / Socket.IO API
  -> Local dashboard
```

| Part | Technology | Purpose |
| --- | --- | --- |
| Backend | Flask, Flask-SocketIO | HTTP API and real-time updates |
| Video | OpenCV | Webcam capture and MJPEG streaming |
| Person tracking | Ultralytics YOLO11n + ByteTrack | Detect and retain person IDs |
| Action model | PyTorch MobileNetV3 + LSTM | Classify a short clip as NonFight/Fight |
| Dashboard | Flask template or optional React app | Display live feed, state, and alerts |

## Run locally

From the project root on Windows PowerShell:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision
python app.py
```

Open <http://localhost:5000>. The first launch may download `yolo11n.pt`.

The dashboard runs without the action-model file, but possible-violence scoring
is disabled until a compatible checkpoint is installed at:

```text
backend/models/violence_action_model.pt
```

## Current trained-model baseline

The table below describes the **original CNN/LSTM baseline**, which remains
compatible with the current runtime. The selected offline experiment is now
MobileNetV3 features + mean/std pooling + logistic regression (`C=0.01`):
75.74% validation F1 and 76.00% accuracy / 78.89% precision / 71.00% recall /
74.74% F1 on the 400-clip test set. It is saved in Drive and **not yet integrated
into the dashboard**. PCA/KNN reached 74.56% validation F1 and was not tested.

See [the complete experiment record](ML_TRAINING_EXPERIMENT.md) and
[the main README's next steps](README.md#next-steps) for artifact locations,
deployment work, and evaluation limitations.

The current baseline was trained in Google Colab on RWF-2000 with the official
`val` split held out for final testing. The configuration used a pretrained
MobileNetV3 feature extractor, one LSTM layer with hidden size 64, dropout,
AdamW weight decay, and a 16-frame clip.

| Selected epoch | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 66.3% | 70.5% | 68.8% | 75.0% | 71.8% |

The run was allowed a maximum of 10 epochs and stopped at epoch 5 through
early stopping. The epoch-2 checkpoint was selected by validation F1, not by
the final epoch number. See [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md)
for the complete record and limitations.

## Runtime behaviour

- A rolling clip uses the `sequence_length` stored inside the checkpoint.
- The action score is evaluated about every 16 processed frames.
- The application averages the five latest action scores.
- A `POSSIBLE_VIOLENCE` alert is created when that average reaches the
  checkpoint threshold (0.80 for the current baseline).

The person-detection threshold in the UI/API is separate from the
violence-action threshold saved in the checkpoint.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Dashboard page |
| `GET /video_feed` | Annotated MJPEG stream |
| `GET /api/detection` | Current tracking, model, and alert data |
| `GET /api/alerts?limit=10` | Recent alerts |
| `POST /api/threshold` | Set YOLO person-detection confidence threshold |
| `POST /api/alerts/<id>/acknowledge` | Mark an alert reviewed |

## Safety and privacy

- Treat every alert as a prompt for human review, not evidence of violence.
- Obtain authorisation before recording or processing video.
- Do not expose this prototype directly to the internet; it has no production
  authentication, authorisation, audit trail, or secure persistent storage.
- Evaluate on relevant, authorised footage before any real-world use.

## Documentation

- [Main README](README.md)
- [Training guide](backend/TRAINING_GUIDE.md)
- [Colab quick start](backend/COLAB_QUICKSTART.md)
- [Training experiment record](ML_TRAINING_EXPERIMENT.md)
