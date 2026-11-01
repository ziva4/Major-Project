# Violence-Action Model Details

This document describes the current PyTorch action-classification model. It
replaces the earlier TensorFlow/Keras synthetic-data demo model, which is not
used by the present dashboard workflow.

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

KNN is not used. KNN compares a sample with nearby stored examples; it does
not learn a time-aware visual representation of the video. The CNN + LSTM
architecture is used because it can learn frame-level visual features and their
movement over time.

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
