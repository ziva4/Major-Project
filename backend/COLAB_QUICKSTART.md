# Google Colab quick start

## Latest experiment status (1 November 2026)

The original baseline workflow below is historical context. Later experiments
used a Tesla T4, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and OpenCV
4.14.0. The shared project folder was added as a shortcut in My Drive, then
mounted at `/content/google_drive/MyDrive/ViolenceDetection`.

Videos and backend were copied to `/content/ViolenceDetectionImproved/`.
All 1,600 training/validation videos passed decode checks. A sorted, seeded
1,440/160 split was saved as `experiments/improved_v1/split_manifest.json` and
reused for later comparisons. Official `val` remained the 400-clip test set.
Use `num_workers=0` for the recorded workflow; local Colab copies disappear
when the runtime resets, while Drive artifacts persist.

The selected offline method is cached MobileNetV3 temporal mean/std features +
logistic regression: 75.74% validation F1, 76.00% test accuracy, and 74.74%
test F1. A separate PCA/KNN comparison reached 74.56% validation F1.
See [the full experiment record](../ML_TRAINING_EXPERIMENT.md) for paths,
all model configurations and the pending full-notebook export. The final bundle
was exported and verified locally on 7 November; dashboard integration is complete.
Only the KNN runnable script's Drive save was confirmed; most later code was
run as notebook cells and is not yet checked into this repository.

Dashboard integration of the `.joblib` classifier is deferred. The local app
still expects CNN/LSTM `.pt` checkpoints. The original baseline setup below
does not automatically reproduce the newer experiments or their saved split.

Use Colab with a **T4 GPU** for RWF-2000 training. The dataset and checkpoints
are stored in Google Drive, so they survive Colab runtime resets.

## Permanent Google Drive layout

```text
My Drive/ViolenceDetection/
  RWF-2000/safe_dataset/
    train/Fight/       train/NonFight/
    val/Fight/         val/NonFight/
  project_backup/backend/
  checkpoints/
```

The safe dataset uses short filenames because the original archive contains a
small number of paths that exceed Colab's filename-length limit.

## Start a new runtime

1. Select **Runtime -> Change runtime type -> T4 GPU**.
2. Mount Drive and restore the backend:

```python
from google.colab import drive
drive.mount('/content/google_drive')

from pathlib import Path
import shutil

DRIVE_ROOT = Path('/content/google_drive/MyDrive/ViolenceDetection')
DATASET = DRIVE_ROOT / 'RWF-2000/safe_dataset'
BACKEND_BACKUP = DRIVE_ROOT / 'project_backup/backend'
LOCAL_BACKEND = Path('/content/MajorProject/backend')

assert DATASET.exists(), f'Missing dataset: {DATASET}'
assert BACKEND_BACKUP.exists(), f'Missing backend backup: {BACKEND_BACKUP}'

LOCAL_BACKEND.parent.mkdir(parents=True, exist_ok=True)
if LOCAL_BACKEND.exists():
    shutil.rmtree(LOCAL_BACKEND)
shutil.copytree(BACKEND_BACKUP, LOCAL_BACKEND)

%pip install -q torch torchvision opencv-python tqdm
%cd /content/MajorProject/backend
```

3. Check the expected 2,000-video dataset:

```python
for relative in ('train/Fight', 'train/NonFight', 'val/Fight', 'val/NonFight'):
    print(relative, len(list((DATASET / relative).glob('*.avi'))))
```

Expected counts are 800, 800, 200, and 200.

## Baseline configuration used in this project

```text
Framework: PyTorch
Architecture: pretrained MobileNetV3-small + 1-layer LSTM
Sequence length: 16 frames
Image size: 160 x 160
LSTM hidden size: 64
Batch size: 4
Optimiser: AdamW (learning rate 1e-4, weight decay 1e-4)
Regularisation: frozen CNN backbone + dropout + early stopping
```

The reported run split the official RWF-2000 training folders into 1,440
training clips and 160 validation clips. The official `val` folders were kept
as the 400-clip held-out test set. Save the best validation-F1 checkpoint to
Drive, then download it to `backend/models/violence_action_model.pt` on the
local computer.

See [ML_TRAINING_EXPERIMENT.md](../ML_TRAINING_EXPERIMENT.md) for the actual
metrics, checkpoint selection, and experiment limitations.
