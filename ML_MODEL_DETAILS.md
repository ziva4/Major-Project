# Violence-Action Model Details

This document describes the PyTorch CNN/LSTM model supported by the current
dashboard runtime. It replaces the earlier TensorFlow/Keras synthetic-data
demo, which is not used by that runtime.

## Selected offline classifier (1 November 2026)

The latest selected experiment replaces the LSTM with temporal mean/std
pooling and logistic regression:

```text
16 uniformly sampled frames -> RGB resize 160 x 160 -> ImageNet normalisation
-> frozen ImageNet MobileNetV3-small -> 576 features per frame
-> temporal mean + population standard deviation -> 1,152 features per video
-> fitted StandardScaler -> logistic regression (C=0.01) -> Fight score
```

It uses original ImageNet CNN weights, not the later fine-tuned CNN. Frame
order is discarded by pooling. The selected scaler/classifier is a `.joblib`
pipeline; a matching feature extractor and preprocessing metadata are required
for inference. It is **not yet integrated into the dashboard**.

| Method | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Follow-up frozen CNN + LSTM-64 | 66.27% | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| Follow-up frozen CNN + LSTM-32 | 67.04% | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| Limited fine-tuning + LSTM-32, best newly trained epoch reported | 66.3% | Not evaluated | Not evaluated | Not evaluated | Not evaluated |
| Selected logistic regression | **75.74%** | **76.00%** | **78.89%** | **71.00%** | **74.74%** |
| PCA + KNN comparison | 74.56% | Not evaluated | Not evaluated | Not evaluated | Not evaluated |

The KNN pipeline uses training-fitted StandardScaler and PCA (32 components),
then 11-neighbour, distance-weighted Euclidean voting. It reuses the same
cached 1,152-dimensional features. Logistic regression remains selected.

These follow-up methods share a newly saved 1,440/160 split; the original
baseline's exact validation membership could not be recovered. The test set
contains 400 official `val` clips. All reported classification metrics use
threshold 0.5; that threshold is not a validated live alert policy.
See [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md) for histories,
artifact paths, limitations, and deferred runtime integration.

## Existing CNN/LSTM runtime

## Purpose

The model classifies a short video clip as either:

| Class value | Label | Meaning |
| ---: | --- | --- |
| 0 | `NonFight` | No fight action in the clip |
| 1 | `Fight` | Fight action in the clip |

It is an educational, binary video-classification model. It does not identify
people, determine intent, or confirm that real-world violence occurred.

## Pipeline

```text
RWF-2000 / webcam video
  -> OpenCV reads video frames
  -> 16 frames are sampled, resized, and normalised
  -> MobileNetV3-small extracts visual features from each frame
  -> LSTM learns temporal movement across the frame sequence
  -> Dropout + linear layer output a Fight probability
```

YOLO11n and ByteTrack are separate components. They detect and track people in
the live dashboard; they are not part of this Fight/NonFight classifier.

## Preprocessing with OpenCV

`train_violence_model.py` uses OpenCV to:

1. open each labelled video clip;
2. sample frames uniformly across the whole clip;
3. resize each selected frame to `160 x 160` pixels;
4. convert BGR frames to RGB;
5. scale pixel values to `0–1` and apply ImageNet normalisation;
6. arrange the result as a tensor with shape:
   `batch x 16 frames x 3 channels x 160 x 160`.

Uniform sampling means each 16-frame input represents the full source clip,
rather than only its first few frames.

## Model architecture

```text
Input: 16 RGB frames, each 160 x 160
        |
        v
MobileNetV3-small feature extractor (pretrained on ImageNet)
        |
        v
Adaptive average pooling -> 576 features per frame
        |
        v
1-layer LSTM, hidden size 64
        |
        v
Dropout (0.35)
        |
        v
Linear layer -> one logit
        |
        v
Sigmoid -> Fight probability from 0 to 1
```

During the completed baseline experiment, the MobileNetV3 feature extractor was
frozen. This limits the number of trainable parameters and helps reduce
overfitting on the available dataset. The LSTM and final classifier learned from
the video labels.

## Training configuration

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| Dataset | RWF-2000 |
| Backbone | MobileNetV3-small, pretrained |
| Temporal model | 1-layer LSTM, hidden size 64 |
| Frames per clip | 16 |
| Frame size | 160 x 160 |
| Batch size | 4 |
| Loss | Binary cross-entropy with logits |
| Optimiser | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Regularisation | frozen backbone, dropout, early stopping |
| Best-checkpoint metric | Validation F1 |

KNN is not part of this CNN/LSTM runtime. It was evaluated separately in the
cached-feature experiment above. The LSTM learns frame-sequence relationships;
KNN and logistic regression instead classify temporal feature summaries.

## Dataset split and metrics

The official RWF-2000 `train` folders provided 1,440 training clips and 160
validation clips using a fixed stratified split. The official `val` folders
were held out as a 400-clip test set.

After every epoch, the project records:

- **Loss**: how different predictions are from true labels;
- **Accuracy**: overall proportion of correct classifications;
- **Precision**: how often a predicted Fight is actually Fight;
- **Recall**: how many actual Fight clips are detected;
- **F1 score**: balance between precision and recall.

The best baseline was epoch 2, selected by validation F1:

| Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: | ---: |
| 66.3% | 70.5% | 68.8% | 75.0% | 71.8% |

See [ML_TRAINING_EXPERIMENT.md](ML_TRAINING_EXPERIMENT.md) for the complete
per-epoch result table, early-stopping decision, checkpoints, and fine-tuning
attempt.

## Runtime use in the dashboard

At runtime, `action_model.py` reads the checkpoint configuration, including the
16-frame sequence length, image size, LSTM size, and alert threshold. The live
application collects a rolling clip, predicts a Fight probability, averages the
latest five scores, and creates a `POSSIBLE_VIOLENCE` alert only when the
average reaches the saved threshold (0.80 in the current baseline).

Every alert must be reviewed by a person. The model's test metrics apply only
to the RWF-2000 split used here and do not guarantee performance on a local
webcam, another camera, or real-world surveillance footage.
