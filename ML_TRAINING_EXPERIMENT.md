# Model Training Experiment Record

This document records the RWF-2000 training work completed for the violence
action classifier. It is a reproducibility record for the final-year project,
not a claim that the system is ready for production use.

## Objective

Compare video classifiers that distinguish RWF-2000 `Fight` and `NonFight`
clips, selecting models using validation F1. The original PyTorch CNN/LSTM
runtime remains in the local dashboard. The newly selected offline model is
MobileNetV3 feature pooling + logistic regression; integration is pending.

## Latest outcome (1 November 2026)

Logistic regression on cached MobileNetV3 temporal mean/std features is the
current selected offline method: **75.74% validation F1; 76.00% test accuracy,
78.89% precision, 71.00% recall, and 74.74% F1**. PCA/KNN achieved 74.56%
validation F1 and was not evaluated on the test set.

The original baseline and its historical fine-tuning attempt are recorded
first. The later Colab experiments, saved artifacts, and pending work follow
below. Results and saves for those experiments are based on Colab output
shared by the user; the Drive contents were not independently inspected.

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

## Lessons from the original baseline

- More epochs are not automatically better; validation F1 determines model
  selection.
- A temporary validation-accuracy decrease is normal, but persistent failure
  to improve validation F1 alongside improving training metrics signals
  overfitting.
- These observations motivated the augmentation, scheduler, smaller-LSTM,
  limited-fine-tuning, and cached-feature experiments recorded below.
- Evaluate the final dashboard on separately collected, authorised local video
  clips. RWF-2000 performance does not guarantee performance on webcam footage.

## Follow-up Colab setup and split

The shared `ViolenceDetection` folder was made accessible by adding a shortcut
in My Drive. Colab mounted it at
`/content/google_drive/MyDrive/ViolenceDetection`; paths containing
`.shortcut-targets-by-id` resolve the same shared-folder shortcut.

- GPU: Tesla T4 (approximately 15 GB).
- Reported packages: PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130,
  OpenCV 4.14.0.
- Videos and backend were copied to `/content/ViolenceDetectionImproved/`
  to avoid repeated reads from Drive. Local Colab files are temporary.
- All 1,600 training/validation videos passed sequential decode checks.
  No test videos were included in this preliminary check.
- DataLoaders used batch size 4 and `num_workers=0`.

The original notebook used seeded shuffling but collected files without
sorting and did not save a split manifest. Its exact validation membership
could not be verified. A **new fixed split** sorted relative filenames before
per-class shuffling with seeds `42 + label`, then reserved 80 clips per class
for validation. Training and validation lists were shuffled with seed 42.
The manifest was saved and reused for all follow-up methods:

| Split | NonFight | Fight | Total |
| --- | ---: | ---: | ---: |
| Training | 720 | 720 | 1,440 |
| Validation | 80 | 80 | 160 |
| Official `val`, retained for testing | 200 | 200 | 400 |

There was no file-path overlap between these splits. This check does not
establish absence of duplicate scenes or event-level leakage.
Original versus follow-up validation scores are not directly comparable.

## Follow-up methods 1 and 2: frozen CNN + LSTM

Both used ImageNet-pretrained MobileNetV3-small, one LSTM layer, dropout 0.35,
AdamW (`lr=1e-4`, `weight_decay=1e-4`), gradient clipping at norm 1.0, and at
most 10 epochs. The CNN parameters and BatchNorm running statistics were
frozen. Inspection of the original Colab notebook confirmed it also called
`model.cnn.eval()`; BatchNorm drift was not established as the baseline cause.

Training sampled one frame randomly from each of 16 temporal segments and
applied clip-consistent horizontal flips (probability 0.5), brightness offsets
in [-0.05, 0.05], and contrast factors in [0.9, 1.1]. Validation used fixed
uniform endpoint sampling without augmentation. Frames were RGB, 160 x 160,
with ImageNet normalisation.

`ReduceLROnPlateau` monitored validation loss, using factor 0.5, patience 2,
and minimum LR `1e-6`. Best checkpoints used validation F1 at probability
threshold 0.5; early stopping patience was 5. This offline threshold is
separate from the existing dashboard's 0.80 alert threshold.

### Per-epoch results

| Method | Epoch | Train loss | Train F1 | Validation loss | Validation accuracy | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: LSTM-64 | 1 | 0.6675 | 56.8% | 0.6550 | 65.0% | **66.27%** |
| 1: LSTM-64 | 2 | 0.6009 | 70.1% | 0.6376 | 65.6% | 63.1% |
| 1: LSTM-64 | 3 | 0.5488 | 73.2% | 0.6502 | 61.3% | 56.3% |
| 1: LSTM-64 | 4 | 0.5158 | 75.5% | 0.6505 | 63.7% | 61.3% |
| 1: LSTM-64 | 5 | 0.4904 | 77.5% | 0.6819 | 63.1% | 57.6% |
| 1: LSTM-64 | 6 | 0.4677 | 79.0% | 0.6903 | 66.2% | 63.0% |
| 2: LSTM-32 | 1 | 0.6678 | 60.5% | 0.6658 | 63.1% | **67.04%** |
| 2: LSTM-32 | 2 | 0.6280 | 67.6% | 0.6521 | 62.5% | 62.5% |
| 2: LSTM-32 | 3 | 0.5804 | 71.9% | 0.6457 | 63.1% | 63.4% |
| 2: LSTM-32 | 4 | 0.5497 | 74.0% | 0.6566 | 60.0% | 56.8% |
| 2: LSTM-32 | 5 | 0.5380 | 75.1% | 0.6647 | 61.3% | 60.8% |
| 2: LSTM-32 | 6 | 0.5186 | 76.0% | 0.6526 | 63.7% | 64.6% |

Both stopped after epoch 6 and selected epoch 1. Training improvement without
sustained validation improvement suggested overfitting; shrinking the LSTM
alone did not resolve it. Method 2 had 78,113 trainable parameters versus
164,417 in method 1. Its 0.77 percentage-point validation-F1 increase is not
evidence of a reliable advantage from one seed and 160 validation clips.

Method 1 was interrupted after epoch 1 and successfully resumed from its
latest checkpoint. Latest checkpoints included optimizer/scheduler state,
history, split membership, and random-generator states.

Scheduler/early-stopping interaction limited the comparison: method 1 trained
only epoch 6 at the reduced `5e-5` rate; method 2's rate was reduced at the end
of epoch 6, when it stopped, so it never trained at that reduced rate.
Neither follow-up LSTM model was evaluated on the test set.

## Follow-up method 3: limited CNN fine-tuning + LSTM-32

This later experiment is distinct from the historical fine-tuning attempt.
It started from method 2's epoch-1 best checkpoint, whose validation F1 was
rechecked at 67.04% before updating weights.

- Only MobileNet feature blocks `11` and `12` were unfrozen.
- All BatchNorm parameters and running statistics remained frozen.
- Trainable CNN parameters: 346,896; LSTM/classifier: 78,113.
- CNN LR: `1e-5`; head LR: `5e-5`; weight decay: `1e-4`.
- Same split, augmentation, frame preprocessing, and batch size as methods 1/2.
- Maximum: 10 additional epochs; scheduler factor 0.5 and patience 1.
- Early stopping required 5 epochs without improved validation F1 and at least
  2 completed epochs at reduced rates, subject to the 10-epoch cap.
- The starting checkpoint was saved as the fallback best model.

| Fine-tuning epoch | Train loss | Train accuracy | Train F1 | Validation loss | Validation accuracy | Validation precision | Validation recall | Validation F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6190 | 69.6% | 70.3% | 0.6533 | 63.7% | 62.0% | 71.2% | **66.3%** |
| 2 | 0.5993 | 70.0% | 71.3% | 0.6420 | 61.9% | 61.7% | 62.5% | 62.1% |
| 3 | 0.5641 | 73.6% | 74.3% | 0.6334 | 62.5% | 62.2% | 63.7% | 63.0% |
| 4 | 0.5390 | 73.8% | 74.6% | 0.6375 | 62.5% | 62.8% | 61.3% | 62.0% |

The last shared output showed epoch 5 starting. No final completion or stop
output was supplied. The user found roughly 10 minutes per epoch too costly
without improved validation F1, and the work moved to cached-feature methods.
Do not claim a completed 10-epoch run or an automatically triggered stop.
Among the reported epochs, no fine-tuned model beat the starting checkpoint.
No test metrics were recorded for this method.

## Cached-feature method: MobileNetV3 + logistic regression

To avoid repeating video decoding and CNN extraction each epoch, the next
experiment used a fresh **ImageNet-pretrained, frozen MobileNetV3-small**,
not the fine-tuned weights. Both training and validation used deterministic,
unaugmented, uniformly sampled 16-frame clips with the same RGB resize and
normalisation as validation above.

The CNN produced 576 features per frame. Temporal mean and population standard
deviation (`unbiased=False`) were concatenated into 1,152 features per video.
This summarises appearance and feature variation but discards frame order;
it does not explicitly learn a movement sequence like an LSTM.

Features were saved in 80-clip chunks (18 training, 2 validation), with
metadata/signature checks. StandardScaler was fitted on training features
only. Logistic regression used `lbfgs`, `max_iter=5000`, seed 42, and the
following validation-based search. Prediction threshold remained 0.5.

| C | Train loss | Train accuracy | Train F1 | Validation loss | Validation accuracy | Validation precision | Validation recall | Validation F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0001 | 0.5258 | 74.8% | 74.5% | 0.5854 | 68.8% | 70.3% | 65.0% | 67.5% |
| 0.001 | 0.4107 | 82.0% | 81.7% | 0.5272 | 71.2% | 71.2% | 71.2% | 71.2% |
| **0.01** | 0.2456 | 92.5% | 92.5% | 0.5363 | **74.38%** | **71.91%** | **80.00%** | **75.74%** |
| 0.1 | 0.0925 | 99.3% | 99.3% | 0.7668 | 73.8% | 71.1% | 80.0% | 75.3% |
| 1.0 | 0.0206 | 100.0% | 100.0% | 1.3551 | 70.6% | 68.1% | 77.5% | 72.5% |

`C=0.01` was selected by validation F1. Its 8.70 percentage-point validation-F1
gain over LSTM-32 is a same-split observation, not proof of a universal
advantage. The training/validation gap remains; overfitting was not eliminated.
Cached classifier searches were much faster than repeated CNN/LSTM epochs.

### Logistic-regression test evaluation

The saved classifier was evaluated without refitting on the 400 official
`val` clips. Matching features were cached in 5 additional chunks.

| Loss | Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.5188 | **76.00%** | **78.89%** | **71.00%** | **74.74%** |

| Actual class | Predicted NonFight | Predicted Fight |
| --- | ---: | ---: |
| NonFight | 162 | 38 |
| Fight | 58 | 142 |

This correctly classified 304/400 clips. Relative to the original baseline's
reported results, accuracy and F1 increased but recall decreased (75% to 71%).
The original internal validation split differed, and the official test set had
already been inspected in earlier baseline runs. Treat this as a held-out
evaluation for the selected classifier, not a newly unseen independent study.

## KNN comparison on the same cached features

All KNN work was saved separately under `ViolenceDetection/KNN_Experiment`.
Only training and validation caches were copied for model selection.

Pipeline: **StandardScaler -> PCA -> distance-based KNN**. Scaling and PCA
were fitted only on training features. A saved script evaluated 24 combinations:
PCA components {32, 64, 128}, neighbours {3, 5, 11, 21}, and voting weights
{uniform, distance}. Euclidean distance and brute-force search were used.
Selection used validation F1 at threshold 0.5; equal scores retained the first
candidate. The 24-configuration search took about 3 seconds in the shared log.

| Selected PCA components | Neighbours | Voting | Validation loss | Accuracy | Precision | Recall | F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 11 | Distance-weighted | 0.7217 | 73.12% | 70.79% | 78.75% | **74.56%** |

KNN was 1.18 percentage points below logistic regression on validation F1.
This is a small single-split difference, not a decisive general ranking.
Logistic regression remains selected. **KNN was not evaluated on the test set.**
Repeated configuration searches also adapt choices to this validation set;
avoid treating the best observed validation score as an unbiased final metric.

## Saved follow-up artifacts

Paths below are relative to `My Drive/ViolenceDetection/` and reflect the
successful Colab save outputs shared in this session.

```text
experiments/
  improved_v1/
    split_manifest.json
    video_decode_report.json
    stage1_best.pt                 # LSTM-64, epoch 1
    stage1_latest.pt               # epoch 6
    stage1_history.json
  lstm32_v1/
    split_manifest.json
    stage1_best.pt                 # LSTM-32, epoch 1
    stage1_latest.pt               # epoch 6
    stage1_history.json
  lstm32_finetune_v1/
    split_manifest.json
    finetune_best.pt               # starting model in last reported output
    finetune_latest.pt             # latest completed epoch; last reported: 4
    finetune_history.json
  mobilenet_logistic_v1/
    feature_cache/                # train, validation, and test .npz chunks
    feature_metadata.json         # preprocessing, versions, exact split
    best_logistic_classifier.joblib  # fitted scaler + classifier
    classifier_results.json
    test_results.json
    test_predictions.npz
KNN_Experiment/
  README.md
  feature_metadata.json
  features/                       # training and validation caches only
  scripts/train_knn.py
  models/best_knn.joblib           # fitted scaler + PCA + KNN
  results/
    validation_results.json
    validation_predictions.npz
    package_versions.json
    KNN_SUMMARY.md
```

A cell was also supplied to create
`experiments/mobilenet_logistic_v1/final_output/` containing the classifier,
metadata, results, `mobilenet_feature_extractor.pt`, package versions, and
`FINAL_RESULTS.md`. **Its execution was not confirmed.** The current Colab
notebook's save/export status was not confirmed either. Most experiment code
was supplied as notebook cells; only the KNN script's save was confirmed.
These Drive artifacts have not been downloaded into this repository.

## Current decision and next steps

1. Retain MobileNetV3 temporal mean/std features + logistic regression
   (`C=0.01`, offline threshold 0.5) as the selected completed offline method.
2. Verify/create the final output bundle, including the exact feature extractor
   and package versions. Save/export the Colab notebook and preserve runnable
   feature extraction, selection, and evaluation code in the repository.
3. Later, download the selected artifacts and implement a backend runtime
   wrapper for CNN features, mean/std pooling, and the fitted scaler/classifier.
   Existing `action_model.py` loads CNN/LSTM `.pt` checkpoints only; renaming
   the `.joblib` file does not deploy the new algorithm. KNN needs its own
   pipeline if used as an alternative.
4. Integrate model readiness, scores, and human-review alerts into the Flask
   dashboard and optional React frontend; replace stale person-only UI labels
   where appropriate. This work was explicitly deferred by the user.
5. Verify rolling-webcam sampling and preprocessing against training. Uniform
   frames across a source clip and consecutive live frames cover different
   time spans. Validate any smoothing and live alert threshold on separate
   footage; the original 0.80 threshold is not validated for logistic regression.
6. Keep existing test results fixed. If continuing model/threshold selection
   after inspecting them, use a new independent final evaluation set. Test on
   authorised local footage before drawing conclusions about webcam behaviour.

## Ethical and practical limitation

This classifier estimates a class from short video clips. It cannot establish
intent, context, or certainty, and it can fail under different cameras,
lighting, activities, or populations. Keep a human reviewer responsible for
every alert.
