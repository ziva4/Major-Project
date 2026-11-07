# Quick start: current Flask dashboard

Updated **7 November 2026**. Use this guide for the integrated logistic model.
Historical TensorFlow/LSTM auto-training instructions no longer apply.

## Start on this laptop

Run in PowerShell:

```powershell
& "C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project\backend\.venv-inference\Scripts\python.exe" -X utf8 "C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project\backend\app.py"
```

Open **http://127.0.0.1:5000**. Keep the terminal open.
Press Ctrl+C and rerun the command to load code changes.

## Choose a source

- **Start live detection:** opens the configured camera. Allow a full five-second
  window plus inference time for the first score.
- **Upload and detect:** choose MP4/AVI/MOV/MKV/WebM, up to a 256 MB request.
  View one whole-video Fight score and frame-by-frame person tracking.
  Pause/resume/stop are available; CPU processing can slow playback.

The camera stays off until selected. Changing source resets tracking and alert history.

## Create demo alerts

Move the **Settings / Demo alert threshold** slider. This enables demo alerts
at the selected percentage (initial 70%). A score at or above the percentage
creates an entry in Recent Demo Alerts, including a valid completed-upload
score. Repeated above-threshold updates do not flood the list with duplicates.

The slider does **not** control person detection. Fight class prediction uses
50%; the separate person-confidence setting also defaults to 50%.
A 73.9% Fight score creates an alert at 70%, but not at 80%.
No police or other external notification is sent.

## Test clips already on this laptop

Browse to:

`C:\Users\Shiva\OneDrive\문서\Major Project\Major-Project\backend\data\local_checks\more_validation`

| Filename | Recorded local Fight score |
| --- | ---: |
| NonFight_2.avi | 27.3270% |
| NonFight_3.avi | 13.2987% |
| Fight_2.avi | 75.8319% |
| Fight_3.avi | 84.4469% |

At a 70% demo threshold the two Fight clips qualify. These are selected
validation examples, not an independent accuracy test. Another earlier
NonFight clip scored about 74.01%, a documented false positive.

## If something fails

- Use the exact environment executable, not the broken plain `python` alias.
- Install matching packages using `backend/requirements-dashboard.txt`;
  scikit-learn must be 1.6.1 and ByteTrack needs `lap`.
- Check model error/wait messages instead of assuming Unavailable means 0%.
- Confirm the exported bundle exists at `backend/models/mobilenet_logistic/`.
- Close other camera applications or use `--camera 1` if needed.
- Ensure only one server uses port 5000.
- Use port 5000; the older React UI on port 3000 lacks the new controls.

See [README_SYSTEM.md](README_SYSTEM.md) for the handoff and
[backend/LOCAL_INFERENCE.md](backend/LOCAL_INFERENCE.md) for detailed setup,
test commands, clip results, and troubleshooting.
