"""Parses arguments"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import OUTPUT_DIR, get_settings
    from .pipeline import process_inputs
except ImportError:
    from config import OUTPUT_DIR, get_settings
    from pipeline import process_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization and export speaker packs plus a manifest."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more media files or a directory containing supported media files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where diarization outputs should be written.",
    )
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error(
            "--num-speakers cannot be combined with --min-speakers or --max-speakers."
        )

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot be greater than --max-speakers.")

    settings = get_settings()
    hf_token = settings.HUGGINGFACE_ACCESS_TOKEN
    if not hf_token.strip():
        parser.error("HUGGINGFACE_ACCESS_TOKEN is missing. Set it in .env or the shell.")

    results = process_inputs(
        inputs=[Path(item) for item in args.input],
        hf_token=hf_token,
        output_root=Path(args.output_dir),
        use_gpu=not args.cpu,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        use_exclusive=True,
    )

    for result in results:
        summary = result["summary"]
        print(f"Input: {result['input_path']}")
        print(f"Output: {result['output_dir']}")
        print(f"Manifest: {result['manifest_path']}")
        print(
            "Summary: "
            f"{summary['speaker_count']} speakers, {summary['segment_count']} segments"
        )

        for speaker in result["speakers"]:
            print(
                f"  {speaker['speaker_key']}: {speaker['segment_count']} clips -> "
                f"{speaker['pack_path']}"
            )
        print()


if __name__ == "__main__":
    main()
