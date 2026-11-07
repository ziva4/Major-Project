# Current system and future-session handoff

Last updated: **7 November 2026**. This is the current architecture/handoff
reference. Read [README.md](README.md) first for launch and user workflow.
Training chronology belongs in [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md).

## Current decision and user preferences

- Use **MobileNetV3-small temporal mean/std features + logistic regression**.
  Do not restart model training or substitute LSTM/KNN without a new request.
- The built-in Flask dashboard at **http://127.0.0.1:5000** is the active demo.
- Ask for a source through the UI: live webcam or uploaded file.
- The Settings slider must control **demo alert threshold**, not person confidence.
- Initial demo threshold: 70%; moving it enables alerts. Score >= threshold
  qualifies. A 73.9% score at 70% must appear in Recent Demo Alerts.
- Keep changes in existing files where appropriate; avoid duplicate implementations.
- The user prefers the exact full PowerShell command and direct HTTP link.
- No external dispatch/notification, production deployment, or retraining is
  required by the completed work.

The user reported successful dashboard behaviour after the live capture fix
and later source/threshold changes. This is a manual functionality report,
not a measured real-world accuracy or long-duration stability study.

## Architecture

```text
Flask startup -> load YOLO + logistic bundle -> idle (camera off)
  |
  +-- Live source
  |     Capture worker -> newest full-size frame + timed 160 x 160 buffer
  |     Tracking worker -> YOLO11n COCO person class + ByteTrack IDs
  |     Action worker -> 16 time-spaced frames over five seconds
  |                   -> MobileNetV3 -> mean/std -> scaler/logistic -> Fight score
  |
  +-- Uploaded source
        Validate/store local file -> whole-video score from 16 uniform samples
        -> sequential person tracking on all frames -> completed
  |
  +-- /api/detection + Socket.IO + MJPEG -> built-in dashboard
        -> slider enables/rechecks demo policy -> in-memory review alerts
```

## Live mode

Capture, tracking, and action classification are separate threads. This is
essential: reading camera frames inside YOLO's loop previously limited capture
to roughly 2 FPS and prevented 16 distinct frames fitting a five-second window.
Do not reintroduce that coupling.

YOLO consumes the newest frame, skipping backlog. Capture stores resized
action frames in a bounded deque, retaining a bracket around the window start.
The action worker chooses frames nearest 16 equally spaced timestamps,
requires distinct indices/full duration, and rejects excessive capture gaps.
It attempts a prediction at most once per second. There is no inherited
five-score moving average.

Score freshness is measured from the sampled evidence's end timestamp.
Evidence older than three seconds is hidden as stale. Read failure clears the
buffer, displayed detections, and action evidence. A generation counter
invalidates in-flight results after resets. Camera FPS and detector FPS are
reported separately.

Warmup diagnostics distinguish insufficient frames, incomplete time span,
interrupted/sparse capture, and calculation in progress. A loaded model does
not guarantee a currently available score.

## Uploaded mode

- Formats: MP4, AVI, MOV, MKV, WebM; maximum HTTP request size 256 MB.
- Require successful initial decoding, at least 16 reported frames, finite
  positive source FPS. Later decoding errors are surfaced.
- Server-generated filenames under ignored `backend/data/uploads/`; original
  basename is for UI display only. Failed uploads are removed. Accepted files
  remain local; no automatic cloud transfer is implemented.
- One **whole-video** Fight score uses 16 uniform endpoint samples. This is
  identical in scope to the offline predictor, not a rolling or localised score.
- A worker then tracks persons in every frame in source order. Progress counts
  processed frames. CPU inference can make playback slower than real time.
- Pause/resume controls processing between frames; stop releases the source.
- The score remains visible through pause/completion. It is cleared on stop
  or source switch. Live freshness expiry does not invalidate a completed file.
- Source changes are serialized by an app lock. Existing workers must finish
  stopping before a new source reuses models/trackers.
- Switching clears alert history and source evidence and resets tracker state.
  Alert IDs continue increasing in the process.

The camera is not opened on server startup, so uploaded mode is usable even
without a webcam.

## Thresholds and alerts

| Item | Value/behaviour |
| --- | --- |
| Person detector confidence | 0.50 by default; legacy/API control only |
| Offline/class-display threshold | 0.50, unchanged |
| Demo alert slider | 0 to 100%; initial 70% |
| Initial alert enablement | Off unless `--demo-alerts`; moving slider enables it |
| Comparison | Fight score >= selected demo threshold |
| Immediate recheck | Valid current live score or completed/paused upload |
| Exclusions | Missing evidence, stale live score, idle/stopped/error source |
| Deduplication | One entry on crossing into qualifying state, not every update |
| History | In memory, acknowledged through API, cleared on source switch/restart |
| Threshold lifetime | Survives source changes; resets on process restart |

Raising a slider above a score does not delete existing history. Lowering it
again after leaving the qualifying state can create another crossing alert.
Acknowledgement marks review status; it does not change model weights or scores.
Alert records retain the creation threshold and whole-video/rolling scope.

Slider requests use a 200 ms debounce and serialized writes so the latest
selected percentage wins. UI reads the action threshold from
`violence_threshold`, not `confidence_threshold`. Do not wire it back to the
legacy person endpoint.

## Environment, artifacts, and paths

Workspace:
`C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project`.

Use `backend/.venv-inference/Scripts/python.exe`; the ordinary `python`
command on this laptop resolves to a broken WindowsApps alias.
The environment reuses installed CPU packages through system-site-packages,
with isolated scikit-learn 1.6.1. System scikit-learn remains 1.8.0.

| Component | Recorded local version |
| --- | --- |
| Python | 3.13.5 |
| PyTorch / torchvision | 2.9.1+cpu / 0.24.1+cpu |
| scikit-learn | 1.6.1 |
| NumPy / OpenCV | 2.2.6 / 4.12.0 |
| lap | 0.5.14 |

`requirements-inference.txt` covers standalone inference;
`requirements-dashboard.txt` adds Flask/Socket.IO/CORS/YOLO/ByteTrack packages.
Old `requirements.txt` and `run.sh` belong to the previous setup.
The missing `lap` dependency originally caused repeated failed Ultralytics
auto-installs; it is now explicit in dashboard requirements.

Local bundle: `backend/models/mobilenet_logistic/`.
Drive source: `My Drive/ViolenceDetection/experiments/mobilenet_logistic_v1/final_output/`.
The shared Drive is available via a My Drive shortcut; resolved
`.shortcut-targets-by-id` paths refer to that shared folder.

Required inference files: classifier joblib (includes scaler), saved CNN feature
weights, feature metadata, and version JSON. The complete export also contains
classifier search results, final Markdown results, test metrics, and test predictions.
Only load the trusted user's joblib export. CNN state is loaded with
`weights_only=True`, `weights=None` prevents a new ImageNet download, and class
labels/features/selected C are validated.

`backend/data/`, binary weights/joblib/npz, virtual environments, and
`backend/inference_results/` are ignored by Git. Small bundle JSON/Markdown
may appear as untracked metadata. A Git clone alone does not restore all models.

## Verification and evidence

- Three bundle-backed tests passed for preprocessing/pooling, video sampling,
  and input errors.
- Latest integration suite: **24 passed**, covering source selection, uploads,
  pause/resume, score freshness, capture independence, thresholds, and alert deduplication.
- Embedded dashboard JavaScript syntax and Git whitespace checks passed.
- A real-model HTTP upload smoke test used a derived 16-frame AVI and completed
  with actual YOLO/logistic weights without opening a camera.
- Two downloaded validation clips matched original cached Colab feature scores
  within 1e-6. One was a genuine NonFight false positive.
- Four additional selected validation clips matched their labels. These checks
  do not constitute an independent evaluation; exact paths/scores are in the
  experiment and local-inference guides.

Local clips:
`backend/data/local_checks/` and `backend/data/local_checks/more_validation/`.
Results:
`backend/inference_results/local_checks_summary.json` and
`backend/inference_results/more_validation/summary.json`.
Do not claim these ignored files will be available after a fresh clone.

## Current completion and remaining work

Completed: selected model export/download, standalone inference, two-clip
Colab parity check, live worker integration/fix, upload/source chooser, runtime
slider-controlled demo alerts, tests, and detailed local documentation.

Remaining:
1. Independent labelled footage and systematic error/latency/live calibration.
2. Save/export the Colab notebook and runnable experiment scripts into reproducible
   project sources; complete notebook export is not confirmed.
3. Optional React feature parity if needed; it is not the current dashboard.
4. Optional upload performance improvement, durable history, or deployability
   work only if requested; these are not implemented.
5. Report/demo preparation and continued version-control backups.

Repository: `https://github.com/ziva4/Major-Project.git`, branch `main`.
The user authorised committing and pushing the accumulated integration on
7 November. This update includes source, tests, documentation, and small
model metadata; ignored binaries and videos still require the Drive backup.
Use Git history/status for the exact commit and remote synchronisation state.

## Documentation map

- [README.md](README.md): current launch and product behaviour.
- [LOCAL_INFERENCE.md](backend/LOCAL_INFERENCE.md): setup, scores, troubleshooting.
- [API_REFERENCE.md](API_REFERENCE.md): current endpoints; lower legacy examples are historical.
- [ML_MODEL_DETAILS.md](ML_MODEL_DETAILS.md): feature/classifier details and comparisons.
- [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md): dated experimental history.
- [frontend/README.md](frontend/README.md): older React limitations.
