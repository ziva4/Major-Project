# Exported Colab experiment notebook

Saved locally on 7 November 2026 from the user-downloaded Colab export.

- File: [RWF2000_Improved_Training.ipynb](RWF2000_Improved_Training.ipynb)
- Source: https://colab.research.google.com/drive/1g0HOiLWFcsIDBTFWg06ay2AYgtL17bOR
- Original downloaded file preserved byte for byte, including outputs.
- 34 code cells, 33 cells with saved outputs; final cell is empty.
- SHA-256: `27ec16d52ff6632cee21c2eeedd908d4235c15ab65d4e8dc997a2d075f50a722`

## Contents verified

The export contains the saved split workflow, CNN/LSTM runs, LSTM-32 and limited
CNN fine-tuning, MobileNet pooled-feature extraction, logistic-regression
selection and held-out test, KNN comparison, final bundle export/verification,
and validation-video download steps.

Saved logistic outputs match the experiment record: C=0.01, validation F1
75.74%, held-out test accuracy 76.00% and F1 74.74%, TN162/FP38/FN58/TP142.
KNN output records PCA32, 11 distance-weighted neighbours and validation F1
74.56%; KNN has no test evaluation.

## Use as a historical experiment record

This is an interactive notebook containing alternative and repeated steps,
not a cleaned sequential Run all pipeline. It preserves a NameError in cell
index 6 and interrupted runs at indices 14 and 18 (zero-based). Later results
are present; these historical errors are not newly executed failures.
Review cells and dependencies before running. Training, downloads, overwrites
and repeated test evaluation can be triggered by running all cells blindly.
No training was rerun during this export.

Colab uses Drive at `/content/google_drive/MyDrive/ViolenceDetection` and a
local dataset copy; those paths are not Windows paths. Recreate the environment,
mount Drive, inspect saved manifests and caches, and run only the required
workflow. Do not retune against the already reported test set.

The source notebook was saved to Drive by the user before download. This local
export has not additionally been uploaded into a new Drive backup folder.
Large models/datasets remain in Drive, not embedded in this notebook.
See [the full experiment record](../ML_TRAINING_EXPERIMENT.md) and
[current system handoff](../README_SYSTEM.md) for context.
