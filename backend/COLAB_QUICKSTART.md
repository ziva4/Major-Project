# Colab quick start

This trains in Google Colab, so the large dataset and GPU workload do not use
your laptop. RWF-2000 is an academic surveillance-video dataset of 2,000 clips;
the source archive is about 9.4 GB. Review its terms and ensure Google Drive has
at least 20 GB free before starting. [Zenodo dataset record](https://zenodo.org/records/15687512)

1. Upload the `backend` folder to `MyDrive/MajorProject/backend`.
2. In Colab, create a notebook, enable **T4 GPU**, and run these cells:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/MajorProject/backend
!pip install torch torchvision opencv-python
!apt-get update -qq && apt-get install -y -qq p7zip-full
!python download_rwf2000.py --output data/RWF-2000
```

3. The download script prints the exact folder which contains `train/Fight`,
`train/NonFight`, `val/Fight`, and `val/NonFight`. Use that printed path:

```python
!python train_violence_model.py --data <printed-dataset-path> --epochs 15 --batch-size 4
```

`train_violence_model.py` preserves the official RWF-2000 training/validation
split when that layout is present. It saves `models/violence_action_model.pt`.
Download only that checkpoint and put it into the local `backend/models/` folder,
then restart the local dashboard with `./run.bat`.
