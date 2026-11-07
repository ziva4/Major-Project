# Optional React frontend

Last updated: **7 November 2026**.

This directory contains the older optional React dashboard. The working demo
is the **built-in Flask dashboard at http://127.0.0.1:5000**, implemented in
`backend/templates/index.html`. See [the main README](../README.md) for startup.

## What this React source currently does

- Connects to the backend through Socket.IO at localhost:5000.
- Displays the MJPEG feed, highest person confidence, person-presence status,
  processed-frame count and person count.
- Polls the alert list and sends acknowledgement requests.
- Its older slider still calls `POST /api/threshold` for person confidence.
  This is different from the current Flask UI's **demo alert** slider.

## What is not implemented here yet

The current React components do not expose live/upload source selection,
upload progress, pause/resume/stop controls, logistic Fight-score/readiness
states, or the new demo-alert slider. Do not claim feature parity.

The backend now starts idle and does not open the camera automatically. This
React app cannot start a source itself until those controls are added; a source
can be selected using the built-in Flask dashboard for development.

The React frontend does not load/train the ML model. Logistic inference is in
`backend/logistic_model.py`, called by the backend processor. The old instruction
to deploy an LSTM `.pt` checkpoint does not apply to the current backend.

## Development

Start the backend using the environment and command in
[LOCAL_INFERENCE.md](../backend/LOCAL_INFERENCE.md). Then:

```powershell
cd frontend
npm install
npm start
```

React serves at http://localhost:3000 and currently hardcodes backend URLs.
Its build/test scripts are in `package.json`; no React build was run for the
recent backend/Flask-template work.

## If React parity is requested

Use [API_REFERENCE.md](../API_REFERENCE.md) to add source selection/upload and
playback controls. Display person confidence separately from Fight score.
Handle null scores and idle/warmup/stale/error states explicitly. Wire the
demo slider to `POST /api/alert-threshold` and `violence_threshold`; never use
person confidence for that control. Keep whole-video upload scores distinct
from five-second live scores.

The user has not asked to replace the working Flask dashboard with React.
