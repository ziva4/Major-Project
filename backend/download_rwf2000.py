"""Download and extract RWF-2000 in Google Colab, not on the local laptop.

The source archive is about 9.4 GB. Run this only in a Colab/Drive workspace
with sufficient free storage and an authorised research-use case.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from tqdm.auto import tqdm

ZENODO_RECORD = "https://zenodo.org/api/records/15687512"


def download_with_progress(url, target, expected_size):
    """Download one archive part while displaying bytes, speed, and ETA."""
    with tqdm(
        total=expected_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=target.name,
    ) as progress:
        previous_bytes = 0

        def report_hook(block_number, block_size, total_size):
            nonlocal previous_bytes
            downloaded_bytes = block_number * block_size
            progress.update(max(0, downloaded_bytes - previous_bytes))
            previous_bytes = downloaded_bytes

        urlretrieve(url, target, reporthook=report_hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/RWF-2000")
    args = parser.parse_args()
    output = Path(args.output)
    downloads = output / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    with urlopen(ZENODO_RECORD) as response:
        files = json.load(response)["files"]
    # Zenodo returns file metadata as dictionaries. Sort by filename, not the
    # dictionaries themselves (which Python cannot order).
    parts = sorted(
        (file for file in files if file["key"].startswith("RWF-2000.7z.")),
        key=lambda file: file["key"],
    )
    if not parts:
        raise RuntimeError("Could not find RWF-2000 archive parts in the Zenodo record.")
    for file in parts:
        target = downloads / file["key"]
        if target.exists() and target.stat().st_size == file["size"]:
            print(f"Already downloaded: {target.name}")
            continue
        print(f"Downloading {target.name} ({file['size'] / 1024**2:.0f} MB)")
        download_with_progress(file["links"]["self"], target, file["size"])

    seven_zip = shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("Install p7zip first: apt-get update && apt-get install -y p7zip-full")
    first_part = downloads / parts[0]["key"]
    extracted = output / "extracted"
    extracted.mkdir(exist_ok=True)
    subprocess.run([seven_zip, "x", str(first_part), f"-o{extracted}", "-y"], check=True)
    print(f"Extracted dataset to {extracted}")
    dataset_root = next(
        (path.parent.parent for path in extracted.rglob("Fight")
         if path.is_dir() and path.parent.name == "train"),
        None,
    )
    if dataset_root:
        print(f"Train with: python train_violence_model.py --data {dataset_root}")
    else:
        print("Use that folder with: python train_violence_model.py --data <extracted-folder>")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)
