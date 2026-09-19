# Training the violence-action model

The app's YOLO model detects and tracks people. The optional CNN/LSTM model
scores a **short video clip** as `normal` or `possible violence`; it does not
identify people and should always be used with human review.

## 1. Prepare the videos

Create this structure. Use video clips, not individual images.

```text
backend/data/
  normal/
    walking_001.mp4
    sports_001.mp4
  violent/
    staged_fight_001.mp4
```

Use varied lighting, camera angles, distances, crowds, and movement. Include
hard normal examples such as exercise, dancing, sports, hugging, and running.
Keep clips from the same event in the same train/validation split to avoid
misleading results. Only use footage you are authorised to process.

## 2. Train in Google Colab

Upload the `backend` folder (including `data`) to Google Drive, open a Colab
notebook with GPU enabled, and run from the uploaded `backend` directory:

```python
!pip install torch torchvision opencv-python
!python train_violence_model.py --data data --epochs 15 --batch-size 4
```

The resulting file is `models/violence_action_model.pt`. Copy that one file to
the same `backend/models/` folder on this computer and restart `run.bat`.

## 3. Runtime behaviour

The live app keeps a rolling 32-frame clip and runs the CNN/LSTM every 16
frames. It averages the last five scores and creates a `POSSIBLE_VIOLENCE`
alert only when the average reaches the checkpoint threshold (default: 80%).
Tune this only after testing with separate, unseen videos.
