# React Dashboard

This optional React frontend displays data supplied by the local Flask /
Socket.IO backend. The backend performs person tracking and, when its PyTorch
checkpoint is installed, possible-violence scoring. The frontend does not train
or load the machine-learning model itself.

## Start development

Start the backend first:

```powershell
cd backend
python app.py
```

Then start React in a second terminal:

```powershell
cd frontend
npm install
npm start
```

Open <http://localhost:3000>.

## Backend connection

The development frontend expects `http://localhost:5000`. It receives Socket.IO
`detection_update` events and uses the REST endpoints documented in
[../API_REFERENCE.md](../API_REFERENCE.md).

It can show the MJPEG camera feed, person boxes/ByteTrack IDs, current person
count, model readiness, possible-violence score, and alerts.

## Action-model integration

As of 1 November 2026, the selected offline experiment is MobileNetV3 feature
pooling + logistic regression, with 76.00% test accuracy and 74.74% test F1.
It is saved in Google Drive but is **not integrated into the backend or this
frontend**. The instructions below apply only to the existing CNN/LSTM runtime.
The `.joblib` classifier cannot replace the `.pt` checkpoint directly.

Next steps are to add the matching backend feature/classifier wrapper, validate
live clip sampling and alert thresholds, and update the UI's model readiness,
score, and review state. See [the project next steps](../README.md#next-steps)
and [experiment record](../ML_TRAINING_EXPERIMENT.md).

To enable possible-violence scoring, install the selected PyTorch checkpoint at:

```text
../backend/models/violence_action_model.pt
```

The current RWF-2000 baseline uses a 16-frame MobileNetV3 + LSTM checkpoint.
Its runtime configuration is saved inside the `.pt` file and read by the
backend at startup. See [../ML_TRAINING_EXPERIMENT.md](../ML_TRAINING_EXPERIMENT.md)
for the reported metrics.

## Important limitation

The possible-violence score is a model prediction, not confirmation of
violence. Keep human review in the workflow and do not use this prototype for
automated enforcement or high-stakes decisions.
