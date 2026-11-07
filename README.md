# Violence Detection and Person-Tracking Dashboard

Last updated: **7 November 2026**.

A local research/demo application with a working built-in Flask dashboard.
Choose **Live camera** or **Upload video**, view person tracking and a Fight
score, and adjust the demo-alert slider to create reviewable alerts.
The selected classifier is **MobileNetV3-small + temporal mean/std pooling +
StandardScaler + logistic regression**, not the earlier LSTM.

## Start here in a future session

Read this file for current behaviour, [README_SYSTEM.md](README_SYSTEM.md)
for architecture and handoff details, [backend/LOCAL_INFERENCE.md](backend/LOCAL_INFERENCE.md)
for environment/verification, and [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md)
for the full training and integration history. Earlier delivery documents are
marked historical and must not override the current runtime instructions.

No new model training is required to continue integration work. The user has
reported the dashboard working after the capture fix and subsequent changes;
systematic accuracy/latency validation on independent local footage is still pending.

## Launch on this laptop

Paste into PowerShell from any directory:

```powershell
& "C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project\backend\.venv-inference\Scripts\python.exe" -X utf8 "C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project\backend\app.py"
```

Open **http://127.0.0.1:5000**. Leave PowerShell running.
Press Ctrl+C to stop. Restart after code changes; debug reload is disabled.

From the repository root, the shorter equivalent is:

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 backend/app.py
```

`backend/run.bat` is the Windows launcher. Use port **5000** for the current
built-in dashboard; port 3000 is the older optional React frontend.

## How to use the dashboard

1. Choose **Start live detection**, or select a file and **Upload and detect**.
   Server startup loads models but does not open the camera.
2. View person boxes/ByteTrack IDs, person confidence, model state, and Fight score.
3. Move the **Settings slider** to the desired **demo alert threshold**.
   Moving it enables demo alerts and immediately checks a valid existing score.
4. Review entries in **Recent Demo Alerts** and acknowledge them as needed.
5. For uploads, pause/resume processing or stop; choose a new source when ready.

| Mode | Fight score | Video behaviour |
| --- | --- | --- |
| Live camera | 16 frames from a rolling five-second window | Capture, person tracking, and action classification run in separate workers |
| Upload | One score from 16 uniform samples over the entire file | Person tracking processes frames in order, with progress; playback may be slower than real time |

Uploads accept MP4, AVI, MOV, MKV, and WebM with a **256 MB request limit**.
They must be readable with at least 16 frames and valid FPS. Accepted uploads
remain locally in ignored `backend/data/uploads/` with generated filenames.
The video is sent to the local server, not to Drive or an external model.
Upload scores stay visible after playback; they do not localise an event in time.
Source changes reset tracking/clip state and alert history.

### Three different percentages

| Setting | Current default | Meaning |
| --- | --- | --- |
| Fight classification | 50% | Fight vs NonFight class decision |
| Demo alert slider | 70% | Scores at or above this create review alerts when enabled |
| YOLO person confidence | 50% | Filters person detections; no longer controlled by this slider |

Example: a **73.9% Fight score** qualifies for a **70% demo alert**.
Moving the slider rechecks completed-upload scores too. Alerts are generated on
entry into the qualifying state, not repeatedly on every update. Raising the
threshold does not erase history. Alerts/acknowledgements are in memory; they
reset on a source switch or server restart. The chosen threshold survives source
switches but resets to the startup default on restart.

Without a slider change, alerts initially remain disabled unless started with
`--demo-alerts`. Other startup options: `--alert-threshold 0.7`, `--camera 1`.
No police dispatch, email, SMS, or other external notification is implemented.

## Selected model and results

```text
16 frames -> OpenCV RGB resize to 160 x 160 -> ImageNet normalisation
-> frozen ImageNet MobileNetV3-small -> 576 features per frame
-> temporal mean + population std -> 1,152 features
-> saved StandardScaler + logistic regression (C=0.01) -> Fight score
```

| Evaluation | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation, 160 clips | 74.38% | 71.91% | 80.00% | 75.74% |
| Test, 400 clips | 76.00% | 78.89% | 71.00% | 74.74% |

The test confusion matrix was TN=162, FP=38, FN=58, TP=142.
KNN remains an offline comparison (74.56% validation F1; no test evaluation).
Original and follow-up CNN/LSTM configurations are documented in the experiment
record. Do not describe the current classifier as an LSTM or KNN runtime.

## Artifacts and dependencies

The eight-file bundle is installed in `backend/models/mobilenet_logistic/`.
Runtime needs the trusted `best_logistic_classifier.joblib`,
`mobilenet_feature_extractor.pt`, `feature_metadata.json`, and
`package_versions.json`. The joblib already includes StandardScaler.
CNN weights load locally without downloading ImageNet weights.

The separate `backend/.venv-inference/` environment uses scikit-learn **1.6.1**,
matching the export. The laptop uses CPU PyTorch 2.9.1 / torchvision 0.24.1.
Colab used CUDA PyTorch 2.11.0 / torchvision 0.26.0; two real clips matched local
scores within 1e-6. The system scikit-learn 1.8.0 was not replaced.

Install/update dashboard dependencies in the existing environment with:

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 -m pip install -r backend/requirements-dashboard.txt
```

This includes ByteTrack's `lap` dependency. First YOLO launch may download
`backend/yolo11n.pt`. Plain `python` points to a broken Windows alias on this
laptop, so use the explicit environment executable. Older `requirements.txt`
and `run.sh` predate this setup; see the current setup guide before using them.

## Source map

| File | Responsibility |
| --- | --- |
| `backend/app.py` | Flask/Socket.IO, source uploads/controls, threshold endpoints, local startup |
| `backend/video_processor.py` | Separate live workers, upload processing, state, tracking, review alerts |
| `backend/logistic_model.py` | Bundle validation, exact preprocessing/pooling, classifier inference |
| `backend/predict_video.py` | Standalone video-to-JSON command |
| `backend/templates/index.html` | Current source chooser, feed, scores, slider, alert list |
| `backend/test_logistic_model.py` | Three bundle-backed inference tests |
| `backend/test_live_integration.py` | 24 source/state/API/alert integration tests |
| `backend/action_model.py` | Earlier CNN/LSTM definition; retained for training/history |
| `frontend/` | Optional older React source; current feature parity pending |

## Verification and known issues

Three bundle-backed inference tests and 24 integration tests passed at their
recorded checkpoints; embedded JavaScript syntax and Git whitespace checks passed.
Two-clip Colab comparison, real-model upload smoke test, and four additional
validation-clip checks are recorded in [LOCAL_INFERENCE.md](backend/LOCAL_INFERENCE.md).

One earlier **NonFight** clip scored about 74.01%, a false positive reproduced
in Colab. Four additional selected clips classified correctly; this is not an
independent accuracy estimate. Do not adjust thresholds merely to make these
selected clips pass.

If Fight score is unavailable, inspect readiness/wait/error messages. Live mode
needs a complete five-second window; stale scores are hidden. Capture was
separated from YOLO after a 2 FPS detector caused indefinite warmup.
See the troubleshooting guide rather than assuming a model-loading failure.

## Next steps

1. Save/export the current Colab notebook and preserve runnable training/feature
   extraction code; full notebook export is not confirmed.
2. Collect authorised, independent labelled footage; evaluate false alarms,
   missed fights, latency, and live sampling/alert behaviour. A demo threshold
   is not an operationally validated threshold.
3. Improve slow uploaded-video playback if needed; retain clear score scope.
4. Update optional React source controls, model state, score, and alert slider
   if that frontend is needed. The Flask dashboard is the current demo.
5. Prepare report/demo evidence and a reproducible artifact/version manifest.
6. Keep future changes committed and backed up to the existing GitHub repository.
   The 7 November integration update includes code, tests, documentation, and
   small bundle metadata; model binaries and videos remain outside Git.

This prototype has no authentication, durable alert store, or production
hardening. It binds to localhost. Every score/alert requires human review.

## Further documentation

- [System handoff](README_SYSTEM.md)
- [Quick start](QUICKSTART.md)
- [Local inference and troubleshooting](backend/LOCAL_INFERENCE.md)
- [Current API](API_REFERENCE.md)
- [Model details](ML_MODEL_DETAILS.md)
- [Experiment history](ML_TRAINING_EXPERIMENT.md)
- [Colab workflow](backend/COLAB_QUICKSTART.md)
- [Historical CNN/LSTM training guide](backend/TRAINING_GUIDE.md)
- [Optional React status](frontend/README.md)
