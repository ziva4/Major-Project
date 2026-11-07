"""Score a video using the exported logistic model, without webcam access."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from logistic_model import DEFAULT_BUNDLE, LogisticActionRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a video with at least 16 frames")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    try:
        started = time.perf_counter()
        recognizer = LogisticActionRecognizer(args.bundle, args.device)
        result = recognizer.predict_video(args.video)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        formatted = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(formatted + "\n", encoding="utf-8")
        print(formatted)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
