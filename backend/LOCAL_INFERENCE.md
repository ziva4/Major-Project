# Local logistic-regression video inference

The standalone predictor and built-in Flask dashboard support the selected
Colab classifier. Live camera and uploaded-video UI operation were confirmed
by the user after integration fixes; broader independent live validation and
alert calibration remain outstanding. Current handoff: [README_SYSTEM.md](../README_SYSTEM.md).

## Artifacts

Keep the trusted Colab bundle in `backend/models/mobilenet_logistic/`:

- `best_logistic_classifier.joblib` (contains the fitted StandardScaler)
- `mobilenet_feature_extractor.pt` (original ImageNet CNN feature weights)
- `feature_metadata.json`
- `package_versions.json`

Model loading requires scikit-learn 1.6.1, as recorded by the exported bundle.
The PyTorch CPU build need not use Colab's CUDA build, but prediction parity
must be checked on the same video before treating migration as verified.
The wrapper loads saved CNN weights without a network download.

## Windows environment

The inspected laptop has Python 3.13 accessible through `py -3.13`; its plain
`python` command points to a broken Windows application alias. A local venv
can reuse installed PyTorch while isolating the matching scikit-learn:

```powershell
py -3.13 -m venv --system-site-packages backend/.venv-inference
backend/.venv-inference/Scripts/python.exe -X utf8 -m pip install -r backend/requirements-inference.txt
```

For a standalone clean environment on another machine, install a compatible
torch/torchvision pair as well as `requirements-inference.txt`. This setup is
for video inference only, not installation of the older Flask/YOLO requirements.

## Predict a video (from the repository root)

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 backend/predict_video.py "C:/path/to/clip.mp4" --output backend/inference_results/clip.json
```

The JSON reports the Fight score, class at threshold 0.5, sampled frame indices,
frame count, device, and elapsed time (including load). The video must have at
least 16 frames. Unreadable sampled frames fail explicitly rather than being
silently replaced. No webcam is opened and no alert is sent.

Inference matches 16 uniform endpoint samples, BGR-to-RGB conversion, OpenCV
resize to 160 x 160, ImageNet normalisation, 576 frame features, and concatenated
temporal mean/population standard deviation (1,152 features). This does not
validate the time window or alert threshold for rolling webcam inputs.

## Verification

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 -m unittest discover -s backend -p test_logistic_model.py -v
```

These tests require the exported bundle. They check preprocessing/pooling
against the notebook formula, video frame selection, and invalid inputs.
Synthetic tests do not establish performance on real footage or prove numeric
parity with the original Colab runtime.

Local verification on 7 November 2026: all three tests passed using CPU PyTorch
2.9.1, torchvision 0.24.1, and isolated scikit-learn 1.6.1. System scikit-learn
remained at 1.8.0. The two-clip comparison with saved Colab features below
subsequently passed with absolute score differences below 1e-6.

The user subsequently confirmed the live dashboard worked after the capture
fix. Next: collect separate authorised local videos, document false alarms and
missed events, and assess the alert threshold. Preserve the fixed model settings;
the clip comparison does not validate live alert behaviour.

### First real-video smoke check (7 November 2026)

Two clips from the saved validation manifest were exported from Drive and
scored locally at threshold 0.5:

| Original relative path | Label | Local Fight score | Prediction |
| --- | --- | ---: | --- |
| `train/NonFight/0f76804c04869b44722c.avi` | NonFight | 0.74007621 | Fight (false positive) |
| `train/Fight/8c6bc3fda59678f2292f.avi` | Fight | 0.73904999 | Fight (correct) |

Both videos report 150 frames at 30 FPS. The predictor sampled indices
`[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129, 139, 149]`.
Per-clip JSON and source paths/SHA-256 hashes were saved under the ignored
`backend/inference_results/` folder. The videos are under ignored
`backend/data/local_checks/`.

### Colab/local comparison

The user ran the saved classifier on the original cached Colab validation
features with scikit-learn 1.6.1 and reported:

| Clip | Saved-feature Colab score | Local video score | Absolute difference (reported) |
| --- | ---: | ---: | ---: |
| NonFight | 0.7400753575 | 0.7400762105 | 0.0000008530 |
| Fight | 0.7390498170 | 0.7390499917 | 0.0000001747 |

Both differences are below 1e-6, supporting numerical agreement for these two
clips despite the different CPU/GPU package versions. The NonFight false
positive is reproduced by the saved Colab features; it is not evidence of a
local migration defect. These two clips do not estimate accuracy or prove
agreement for every video. Do not change the threshold merely to correct them.

## Built-in webcam dashboard

Dashboard dependencies were installed into the separate inference environment.
ByteTrack requires `lap`; it is explicitly included in the dashboard requirements.
The first real-camera launch exposed the missing optional dependency and its
repeated failed auto-install attempts. `lap` 0.5.14 was installed into this
environment on 7 November 2026. Restart the app after dependency installation.
To reproduce installation, use `requirements-dashboard.txt` rather than the
older `requirements.txt`, whose NumPy/Flask pins predate this Python environment.

From the repository root:

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 -m pip install -r backend/requirements-dashboard.txt
backend/.venv-inference/Scripts/python.exe -X utf8 backend/app.py
```

Or run `backend/run.bat`. Open http://127.0.0.1:5000 and stop with Ctrl+C.
The server starts with the camera off. Choose **Start live detection** or select
a file and choose **Upload and detect** in the dashboard. Switching sources
stops the previous source and resets its detections, tracking IDs, and alerts.
First launch may download the COCO `yolo11n.pt` person-detection weights.
Use `--camera 1` if a different device index is needed.

Camera capture now runs in its own worker, separately from both YOLO tracking
and action classification. YOLO consumes the newest frame and skips backlog;
it does not limit the rate of frames collected for the action model.
The API/UI report camera capture FPS separately from person-detection FPS.

The action worker samples 16 distinct frames nearest equally spaced timestamps
across a five-second rolling window. It waits for a full window and rejects
long capture gaps; sparse/slow capture may therefore remain in warmup. Frames
are resized to 160 x 160 for bounded buffering. Classification runs in a
separate worker, attempting a new score no more often than once a second, with
no inherited five-score smoothing. Scores older than three seconds of video
evidence are hidden; camera read failures clear the clip and require warmup.
The five-second choice matches the duration of the two checked dataset clips,
but has not yet been validated across local webcam footage.

The UI separates person confidence from the Fight score, model readiness,
warmup/error/stale states, and the offline 0.5 class decision. A missing or
incompatible logistic bundle leaves person tracking available with an error
message. REST polling backs up Socket.IO updates.

The dashboard Settings slider now controls the **demo alert threshold**, not
person detection. Its initial value is 70%. Moving it enables demo alerts and
immediately rechecks the current valid score, including a completed upload's
whole-video score. Scores at or above the selected threshold create a review
alert when entering the qualifying state. Repeated updates while still above
the threshold do not create duplicate entries. Raising the threshold does not
erase historical alerts. Stale live scores and empty/cleared sources do not
create alerts. The setting persists across source switches in this server
session; a restart uses the startup default again.

No CLI flag is needed after adjusting the slider. `--demo-alerts` remains an
option to enable alerts immediately at startup; `--alert-threshold 0.7` selects
the startup value. Person confidence stays at its configured default (50%)
and is not changed by this UI slider. The legacy `/api/threshold` endpoint is
retained for person confidence; the new slider uses `/api/alert-threshold`.
Demo alert thresholds are not validated operational settings. Scores and
alerts require human interpretation; neither establishes that violence occurred.

Camera-free integration checks:

```powershell
backend/.venv-inference/Scripts/python.exe -X utf8 -m unittest discover -s backend -p test_live_integration.py -v
```

The suite grew from 12 to 19 and then 24 passing tests on 7 November 2026,
including source selection and alert-slider behaviour. The dashboard JavaScript
syntax check passed.
These checks do not replace real-camera testing. The optional React frontend
has not yet been updated with the new score/readiness display.

The first live session exposed an indefinite-warmup bug: capture was tied to
YOLO, which the API reported running at approximately 2 FPS. That could not
supply 16 distinct frames in five seconds. Capture was separated into its own
worker, and an added regression test verifies that camera frames, the action
buffer, and the video feed keep advancing while the detector is blocked.
Warmup now reports frame collection/wait reasons instead of a generic message.
A server restart is required to load this fix. The user subsequently confirmed
the live display worked; this confirms functionality, not independent accuracy.

## Uploaded-video mode

The source chooser accepts MP4, AVI, MOV, MKV, and WebM uploads, with a 256 MB
request limit. Files must decode and report at least 16 frames and valid FPS.
Accepted files are saved under ignored `backend/data/uploads/` using generated
filenames; the original name is used only for display. Failed uploads are
removed. Accepted uploads remain locally until manually removed when unused.
Files are sent to the local Flask server, not to Google Drive or a cloud model.

Upload mode computes **one whole-video Fight score**, using 16 uniform samples
across the entire source file, exactly as offline video inference does. This
is not a time-localised fight event score. That score remains visible during
playback and after completion. Person detection/tracking then processes every
video frame in source order; playback slows down when inference cannot keep
up with the source FPS. The dashboard shows frame progress and provides
pause/resume and stop controls. A whole-video score differs from live mode's
rolling five-second score; the UI labels that distinction.

There is no camera auto-start on server launch. Missing camera devices do not
prevent using uploaded-video mode. Source changes are serialized, unfinished
workers cannot be reused, and completed upload scores do not expire merely
because playback is paused or finished. Switching or stopping clears them.
Nineteen camera-free integration tests passed after adding source selection,
upload validation, request limits, playback state, completion, and progress.
