# Violence Detection and Person-Tracking Dashboard

A local, webcam-based surveillance prototype that detects and tracks people in
real time and can optionally flag **possible violence** from a short sequence
of video frames. It is designed for demonstration, research, and supervised
review - not autonomous safety or law-enforcement decisions.

## What it does

- Captures a webcam feed with OpenCV.
- Uses the pretrained **YOLO11n** COCO model to detect people and assign
  persistent ByteTrack IDs.
- Streams an annotated MJPEG video feed and live detection data through a
  Flask/Socket.IO dashboard.
- Keeps an alert history that an operator can acknowledge.
- Optionally loads a locally trained **MobileNetV3 + LSTM** checkpoint to score
  a rolling video clip as normal activity or *possible violence*.

The app works for person detection without the action-model checkpoint. Violence
alerts are enabled only after `backend/models/violence_action_model.pt` is
present and loads successfully.

## Latest experiment status (1 November 2026)

The selected offline classifier is now **frozen MobileNetV3-small features +
temporal mean/std pooling + StandardScaler + logistic regression** (`C=0.01`).
It achieved **76.00% test accuracy, 78.89% precision, 71.00% recall, and 74.74%
F1** on 400 RWF-2000 test clips. Its validation F1 was 75.74%.

Three follow-up CNN/LSTM configurations and a PCA/KNN comparison were also
run in Colab. KNN reached 74.56% validation F1, below logistic regression.
See [the experiment record](ML_TRAINING_EXPERIMENT.md) for all results,
per-epoch histories, split changes, and limitations.

**The local dashboard still loads CNN/LSTM checkpoints.** The new logistic
classifier and KNN pipeline are saved in Google Drive and have not been
integrated into this repository's runtime. A `.joblib` classifier cannot be
renamed to `violence_action_model.pt` to deploy it.

Confirmed experiment artifacts are under `My Drive/ViolenceDetection/experiments/`
and `My Drive/ViolenceDetection/KNN_Experiment/`. A separate `final_output/`
packaging cell was supplied, but its execution has not been confirmed.

### Next steps

1. Verify the final output bundle and save/export the current Colab notebook
   and experiment code; bring reproducible code into the repository.
2. Download the selected classifier, matching feature-extractor weights and
   preprocessing metadata, then add a logistic-regression runtime wrapper.
3. Integrate it with the Flask dashboard and optional React frontend, including
   explicit model readiness, score display, and human-review alerts.
4. Validate rolling-webcam sampling against the training clip preprocessing;
   select alert thresholds on separate validation footage. The offline 0.5
   classification threshold does not establish a live alert threshold.
5. Evaluate on separately collected, authorised footage. Further tuning after
   viewing the existing test results needs a new independent final evaluation.

## Prototype limitations

- A high model score is not evidence that violence occurred. Always require
  timely human review before taking action.
- Model quality depends on the training data and may perform poorly across
  different cameras, lighting, crowds, activities, and demographics.
- Obtain authorization for any video processing and follow applicable privacy,
  security, retention, and notice requirements.
- This project has no authentication, access controls, encrypted storage, or
  production deployment hardening. Do not expose it directly to the internet.

## Quick start

### Requirements

- Python 3.10+ recommended
- A working integrated or USB webcam
- Internet access on the first start so Ultralytics can download `yolo11n.pt`
- Windows: PowerShell or Command Prompt (the included `run.bat` is Windows-only)

From the repository root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision
python app.py
```

`torch` and `torchvision` are required by the action-model runtime, even when
no trained action checkpoint has been added. If you use the Windows launcher,
run `backend\run.bat` after creating the environment; it installs the packages
listed in `requirements.txt` and starts the server.

Open <http://localhost:5000>. Grant your browser/system camera permission when
prompted. The first start downloads the small pretrained YOLO model; subsequent
starts reuse the cached model.

To stop the server, press `Ctrl+C` in its terminal.

## Optional violence-action model

The default dashboard provides people detection and tracking. To enable
possible-violence alerts, train a checkpoint and place it here:

```text
backend/models/violence_action_model.pt
```

For GPU training in Google Colab using RWF-2000, follow
[the Colab quick-start guide](backend/COLAB_QUICKSTART.md). For custom videos,
see [the training guide](backend/TRAINING_GUIDE.md).

At runtime, the recognizer uses the rolling clip length saved in the checkpoint
(16 frames for the current RWF-2000 baseline), scores it about every
16 processed frames, averages the last five scores, and creates an alert only
when the checkpoint's threshold is reached (80% by default).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Dashboard page |
| `GET /video_feed` | Annotated MJPEG camera stream |
| `GET /api/detection` | Current person, tracking, model, and alert state |
| `GET /api/alerts?limit=10` | Recent alerts |
| `POST /api/threshold` | Set the person-detection confidence threshold |
| `POST /api/alerts/<id>/acknowledge` | Mark an alert as reviewed |

`POST /api/threshold` accepts JSON such as `{ "threshold": 0.5 }`. This
threshold controls YOLO person detections; it does not change the trained
violence-action threshold.

The server emits `detection_update` events through Socket.IO roughly every
0.5 seconds. See [API_REFERENCE.md](API_REFERENCE.md) for the original endpoint
reference; some examples there predate the current YOLO/action-model runtime.

## Project layout

```text
backend/
  app.py                    Flask, REST endpoints, and Socket.IO server
  video_processor.py        Webcam capture, YOLO tracking, alert lifecycle
  action_model.py           MobileNetV3 + LSTM checkpoint runtime
  train_violence_model.py   Checkpoint training script
  TRAINING_GUIDE.md         Custom-data training instructions
  COLAB_QUICKSTART.md       RWF-2000/Colab workflow
  requirements.txt          Python dependencies
  templates/index.html      Built-in dashboard
frontend/                   Separate React dashboard source (optional)
API_REFERENCE.md            API and demo reference
```

## Development notes

- The built-in Flask dashboard is served at port `5000`.
- The React source under `frontend/` is a separate development frontend. Run
  `npm install` followed by `npm start` from that directory after the backend is
  running.
- Runtime artefacts such as virtual environments, datasets, and trained model
  weights are intentionally excluded from Git; see `.gitignore`.

## Troubleshooting

- **Camera cannot open:** Close other programs using the camera and check OS
  camera privacy settings. If needed, change `camera_index=0` in
  `VideoProcessor` to another device index.
- **`No module named torch` or `torchvision`:** Activate the backend virtual
  environment and run `pip install torch torchvision`.
- **No possible-violence score:** This is expected until a compatible
  `backend/models/violence_action_model.pt` checkpoint is installed.
- **Port 5000 is already in use:** Stop the conflicting process or change the
  port in `backend/app.py`.

## Documentation

- [API reference](API_REFERENCE.md)
- [Training guide](backend/TRAINING_GUIDE.md)
- [Google Colab quick start](backend/COLAB_QUICKSTART.md)
- [RWF-2000 training experiment record](ML_TRAINING_EXPERIMENT.md)
- [Frontend setup](FRONTEND_SETUP.md)
