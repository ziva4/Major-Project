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
