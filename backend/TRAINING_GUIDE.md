# Training the violence-action model

## Latest experiment status (1 November 2026)

This guide's commands train the existing CNN/LSTM implementation. They do not
reproduce all later notebook experiments. The selected offline method is now
frozen MobileNetV3-small + temporal mean/std pooling + fitted StandardScaler +
logistic regression (`C=0.01`): validation F1 75.74%, test accuracy 76.00%,
precision 78.89%, recall 71.00%, and F1 74.74%.

Three follow-up CNN/LSTM configurations and a PCA/KNN comparison were recorded.
KNN reached 74.56% validation F1; its test metrics were not run. All follow-up
comparisons use the new saved split manifest, not a verified reconstruction of
the original notebook's split. See the [experiment record](../ML_TRAINING_EXPERIMENT.md)
for exact settings, per-epoch results, saved files, and limitations.

Artifacts are in Drive under `ViolenceDetection/experiments/` and
`ViolenceDetection/KNN_Experiment/`. The selected `.joblib` pipeline and matching CNN weights have now been exported,
loaded locally, checked against Colab scores, and integrated into the built-in
Flask dashboard (7 November 2026). These older training commands target the
LSTM, not the selected logistic pipeline. Full later-notebook export remains
unconfirmed. See [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md) for current setup.
Do not retune using the reported test results.

The app's YOLO model detects and tracks people. The optional CNN/LSTM model
scores a **short video clip** as `normal` or `possible violence`; it does not
identify people and should always be used with human review.

## Current RWF-2000 baseline

The completed RWF-2000 baseline is documented in
[ML_TRAINING_EXPERIMENT.md](../ML_TRAINING_EXPERIMENT.md). It used a 16-frame,
pretrained MobileNetV3-small + LSTM model with hidden size 64, a frozen CNN
backbone, AdamW, dropout, and early stopping. Epoch 2 was selected by the best
validation F1 (66.3%); it achieved 70.5% held-out test accuracy and 71.8% test
F1. The model was allowed up to 10 epochs and stopped at epoch 5 because
validation F1 did not improve for three consecutive epochs.

The selected Google Drive checkpoint must be copied locally as:

```text
backend/models/violence_action_model.pt
```

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
!python train_violence_model.py --data data --epochs 15 --batch-size 4 --patience 5
```

The resulting file is `models/violence_action_model.pt`. Copy that one file to
the same `backend/models/` folder on this computer and restart `run.bat`.

After every epoch, the script prints training and validation loss, accuracy,
precision, recall, and F1. It saves a checkpoint only when validation F1
improves and stops after five epochs without an improvement by default. Use
`--patience 0` to disable early stopping.

For a very small experimental dataset, start with the pretrained CNN backbone
frozen and a smaller LSTM rather than removing arbitrary layers:

```python
!python train_violence_model.py --data <dataset-root> --hidden-size 64 --freeze-backbone --patience 5
```

This can reduce overfitting, but it cannot compensate for too few or
unrepresentative clips. Fine-tune the full model only after obtaining a larger,
balanced dataset and a reliable held-out split.

For this safety-oriented binary classification task, choose the checkpoint by
validation F1 (and inspect precision and recall), not validation accuracy alone.
Keep the final test set untouched until model selection is complete.

## 4. Held-out test results

To print final test loss, accuracy, precision, recall, and F1, reserve a
separate set of clips that was not used during training or early stopping:

```text
test/
  NonFight/
  Fight/
```

Then supply that folder only when the training run is complete:

```python
!python train_violence_model.py --data <train-and-validation-root> --test-data <test-folder>
```

The script reloads the checkpoint with the best validation F1 and evaluates it
once on the held-out test clips. Do not use that result to tune layers,
thresholds, epochs, or other settings; doing so turns the test set into another
validation set.

## 3. Runtime behaviour

The live app keeps a rolling clip whose length is read from the checkpoint
(16 frames for the current RWF-2000 baseline) and runs the CNN/LSTM about every
16 processed frames. It averages the last five scores and creates a
`POSSIBLE_VIOLENCE` alert only when the average reaches the checkpoint threshold
(default: 80%). Tune this only after testing with separate, unseen videos.
