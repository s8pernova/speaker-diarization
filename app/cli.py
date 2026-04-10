"""Parses arguments"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from diarize import run_diarization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]

    segments, _ = run_diarization(
        audio_path=Path(args.input),
        hf_token=hf_token,
        use_gpu=not args.cpu,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        use_exclusive=True,
    )

    for segment in segments:
        print(segment)


if __name__ == "__main__":
    main()