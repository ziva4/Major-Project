# Model Training Experiment Record

This document records the RWF-2000 training work completed for the violence
action classifier. It is a reproducibility record for the final-year project,
not a claim that the system is ready for production use.

## Objective

Train a PyTorch video classifier that distinguishes RWF-2000 `Fight` and
`NonFight` clips, then select a checkpoint that generalises best to unseen
clips. The selected model can be installed in the local Flask dashboard as a
human-review aid.

## Dataset and split

RWF-2000 contains 2,000 video clips:

| Source folder | Use in this experiment | Clips |
| --- | --- | ---: |
| `train/Fight` | Training/validation source | 800 |
| `train/NonFight` | Training/validation source | 800 |
| `val/Fight` | Held-out final test set | 200 |
| `val/NonFight` | Held-out final test set | 200 |

We used a fixed seed (`42`) to reserve 10% of each training class for
validation. This produced 1,440 training clips, 160 validation clips, and 400
held-out test clips. The test set was not used to select the best epoch.

The original archive contained some paths too long for the Colab filesystem, so
clips were safely extracted to short filenames while preserving the four class
folders. No video labels were changed.

## Baseline model and training configuration

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| Model | MobileNetV3-small feature extractor + LSTM classifier |
| Pretraining | ImageNet MobileNetV3 weights |
| CNN backbone | Frozen during baseline training |
| LSTM | 1 layer, hidden size 64 |
| Clip length | 16 frames sampled from each video |
| Input image size | 160 x 160 |
| Classifier regularisation | Dropout 0.35 |
| Optimiser | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 |
| Maximum epochs | 10 |
| Selection metric | Validation F1 |
| Early-stopping patience | 3 epochs without validation-F1 improvement |

Training metrics were printed every epoch: loss, accuracy, precision, recall,
and F1. Checkpoints were saved to Google Drive after each completed epoch.

## Baseline results

| Epoch | Split | Loss | Accuracy | Precision | Recall | F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Training | 0.6661 | 60.6% | 61.0% | 58.5% | 59.7% |
| 1 | Validation | 0.6571 | 61.9% | 66.7% | 47.5% | 55.5% |
| 2 | Training | 0.5915 | 70.2% | 70.5% | 69.6% | 70.0% |
| 2 | Validation | 0.6302 | 65.0% | 64.0% | 68.8% | **66.3%** |
| 3 | Training | 0.5320 | 74.2% | 72.8% | 77.2% | 74.9% |
| 3 | Validation | 0.6311 | 61.3% | 62.5% | 56.2% | 59.2% |
| 4 | Training | 0.4747 | 78.0% | 77.9% | 78.2% | 78.0% |
| 4 | Validation | 0.6225 | 65.6% | 68.1% | 58.8% | 63.1% |
| 5 | Training | 0.4367 | 80.9% | 80.6% | 81.4% | 81.0% |
| 5 | Validation | 0.6186 | 65.0% | 65.0% | 65.0% | 65.0% |

The model was configured for up to 10 epochs, but early stopping ended the
baseline after epoch 5. Epochs 3, 4, and 5 did not exceed epoch 2's validation
F1 of 66.3%. The growing training-versus-validation gap also indicated that
continuing the same configuration was likely to overfit.

## Selected checkpoint and held-out test result

Epoch 2 was selected solely because it had the highest validation F1.

| Selected epoch | Test loss | Test accuracy | Test precision | Test recall | Test F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5653 | 70.5% | 68.8% | 75.0% | 71.8% |

Interpretation:

- The model detected 75.0% of Fight clips in this held-out test set (recall).
- Of clips predicted as Fight, 68.8% were actually Fight (precision).
- This is a baseline suitable for demonstrating the dashboard workflow, not a
  validated model for real-world deployment.

## Checkpoints and file locations

During training, these files were saved in Google Drive:

```text
My Drive/ViolenceDetection/checkpoints/
  violence_action_model_best_10epochs.pt   # best validation-F1 model (epoch 2)
  violence_action_model_latest.pt          # latest completed baseline state (epoch 5)
```

The deployment copy should be downloaded and placed at:

```text
backend/models/violence_action_model.pt
```

The `.pt` checkpoint contains model weights and the required runtime
configuration, including the 16-frame sequence length and LSTM hidden size.

## Fine-tuning attempt

After the baseline, we prepared a separate controlled experiment starting from
the epoch-2 checkpoint. It used training-only horizontal flips, unfreezing only
the final two MobileNet blocks, frozen BatchNorm layers, a backbone learning
rate of `1e-5`, and a classifier learning rate of `5e-5`.

The first attempt encountered a Colab/PyTorch multiprocessing DataLoader
cleanup error while reading the Drive-mounted video files. The loaders were
reset to `num_workers=0`, the model was reset to the original epoch-2 baseline,
and the follow-up run was stopped before a complete fine-tuning comparison was
recorded. Therefore, no fine-tuned checkpoint is claimed as an improvement.

## Lessons and next steps

- More epochs are not automatically better; validation F1 determines model
  selection.
- A temporary validation-accuracy decrease is normal, but persistent failure
  to improve validation F1 alongside improving training metrics signals
  overfitting.
- Future experiments should use controlled augmentation, learning-rate
  scheduling, or carefully limited backbone fine-tuning, each compared against
  the epoch-2 baseline without using the held-out test set for tuning.
- Evaluate the final dashboard on separately collected, authorised local video
  clips. RWF-2000 performance does not guarantee performance on webcam footage.

## Ethical and practical limitation

This classifier estimates a class from short video clips. It cannot establish
intent, context, or certainty, and it can fail under different cameras,
lighting, activities, or populations. Keep a human reviewer responsible for
every alert.
