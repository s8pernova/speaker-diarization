"""Orchestrates the diarization workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from .config import OUTPUT_DIR
    from .ffmpeg_utils import cut_audio_clip, extract_audio
    from .manifest import write_manifest
    from .merger import merge_speaker_clips
    from .segmenter import group_segments_by_speaker
except ImportError:
    from config import OUTPUT_DIR
    from ffmpeg_utils import cut_audio_clip, extract_audio
    from manifest import write_manifest
    from merger import merge_speaker_clips
    from segmenter import group_segments_by_speaker

SUPPORTED_INPUT_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".wav"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "input"


def _timestamp_token(seconds: float) -> str:
    return f"{int(round(seconds * 1000)):010d}"


def _rounded(value: float) -> float:
    return round(float(value), 3)


def _relative_to(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _build_run_directory_name(input_path: Path) -> str:
    base_name = _slugify(input_path.stem)
    suffix = input_path.suffix.lstrip(".").lower()
    return f"{base_name}_{suffix}" if suffix else base_name


def resolve_input_paths(inputs: Sequence[str | Path]) -> list[Path]:
    """Expand one or more input files or directories into supported media files."""
    resolved_inputs: list[Path] = []

    for raw_input in inputs:
        input_path = Path(raw_input).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_dir():
            directory_files = sorted(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
            )
            if not directory_files:
                raise FileNotFoundError(
                    f"No supported media files found in directory: {input_path}"
                )
            resolved_inputs.extend(directory_files)
            continue

        if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported input file type: {input_path.suffix or '<no extension>'}"
            )
        resolved_inputs.append(input_path)

    deduped_inputs: list[Path] = []
    seen_paths: set[Path] = set()
    for input_path in resolved_inputs:
        resolved_path = input_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        deduped_inputs.append(resolved_path)

    return deduped_inputs


def process_input_file(
    input_path: str | Path,
    hf_token: str,
    output_root: str | Path = OUTPUT_DIR,
    use_gpu: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_exclusive: bool = True,
) -> dict[str, Any]:
    """Process a single media file into diarized clips, speaker packs, and manifest."""
    input_path = Path(input_path).resolve()
    output_root = Path(output_root).resolve()
    run_directory = output_root / _build_run_directory_name(input_path)
    run_directory.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = run_directory / "source_audio.wav"
    clips_directory = run_directory / "clips"
    speakers_directory = run_directory / "speakers"

    extract_audio(input_path=input_path, output_path=extracted_audio_path)

    try:
        from .diarize import run_diarization
    except ImportError:
        from diarize import run_diarization

    diarized_segments, _ = run_diarization(
        audio_path=extracted_audio_path,
        hf_token=hf_token,
        use_gpu=use_gpu,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        use_exclusive=use_exclusive,
    )

    ordered_segments, grouped_segments = group_segments_by_speaker(diarized_segments)
    speaker_records: list[dict[str, Any]] = []

    for speaker_key in sorted(grouped_segments):
        speaker_group = grouped_segments[speaker_key]
        clip_paths: list[Path] = []

        for segment in speaker_group["segments"]:
            clip_path = (
                clips_directory
                / speaker_key
                / (
                    f"{segment['speaker_clip_index']:04d}_"
                    f"{_timestamp_token(segment['start'])}_"
                    f"{_timestamp_token(segment['end'])}.wav"
                )
            )
            cut_audio_clip(
                source_audio_path=extracted_audio_path,
                output_path=clip_path,
                start=segment["start"],
                end=segment["end"],
            )
            clip_paths.append(clip_path)
            segment["clip_path"] = _relative_to(run_directory, clip_path)

        speaker_pack_path = speakers_directory / f"{speaker_key}.wav"
        merge_speaker_clips(clip_paths=clip_paths, output_path=speaker_pack_path)

        speaker_records.append(
            {
                "speaker": speaker_group["speaker"],
                "speaker_key": speaker_key,
                "segment_count": speaker_group["segment_count"],
                "total_duration": _rounded(speaker_group["total_duration"]),
                "pack_path": _relative_to(run_directory, speaker_pack_path),
                "clips": [
                    {
                        "index": segment["speaker_clip_index"],
                        "path": segment["clip_path"],
                        "start": _rounded(segment["start"]),
                        "end": _rounded(segment["end"]),
                        "duration": _rounded(segment["duration"]),
                    }
                    for segment in speaker_group["segments"]
                ],
            }
        )

    manifest_path = write_manifest(
        run_directory / "manifest.json",
        {
            "source": {
                "input_path": input_path,
                "audio_path": _relative_to(run_directory, extracted_audio_path),
            },
            "options": {
                "use_gpu": use_gpu,
                "use_exclusive": use_exclusive,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
            "summary": {
                "speaker_count": len(grouped_segments),
                "segment_count": len(ordered_segments),
                "total_speech_duration": _rounded(
                    sum(segment["duration"] for segment in ordered_segments)
                ),
            },
            "speakers": speaker_records,
            "segments": [
                {
                    "index": segment["index"],
                    "speaker": segment["speaker"],
                    "speaker_key": segment["speaker_key"],
                    "speaker_clip_index": segment["speaker_clip_index"],
                    "start": _rounded(segment["start"]),
                    "end": _rounded(segment["end"]),
                    "duration": _rounded(segment["duration"]),
                    "clip_path": segment.get("clip_path"),
                }
                for segment in ordered_segments
            ],
        },
    )

    return {
        "input_path": input_path,
        "output_dir": run_directory,
        "audio_path": extracted_audio_path,
        "manifest_path": manifest_path,
        "summary": {
            "speaker_count": len(grouped_segments),
            "segment_count": len(ordered_segments),
        },
        "speakers": [
            {
                "speaker": speaker_record["speaker"],
                "speaker_key": speaker_record["speaker_key"],
                "segment_count": speaker_record["segment_count"],
                "pack_path": run_directory / speaker_record["pack_path"],
            }
            for speaker_record in speaker_records
        ],
    }


def process_inputs(
    inputs: Sequence[str | Path],
    hf_token: str,
    output_root: str | Path = OUTPUT_DIR,
    use_gpu: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_exclusive: bool = True,
) -> list[dict[str, Any]]:
    """Process one or more files from the CLI."""
    input_paths = resolve_input_paths(inputs)
    return [
        process_input_file(
            input_path=input_path,
            hf_token=hf_token,
            output_root=output_root,
            use_gpu=use_gpu,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            use_exclusive=use_exclusive,
        )
        for input_path in input_paths
    ]


def main() -> None:
    """Delegate script execution to the CLI entrypoint."""
    try:
        from .cli import main as cli_main
    except ImportError:
        from cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
