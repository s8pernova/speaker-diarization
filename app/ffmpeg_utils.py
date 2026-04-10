"""Extracts audio and cuts clips."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1


def ensure_ffmpeg_installed() -> None:
    """Raise a helpful error when ffmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")


def run_ffmpeg_command(args: list[str]) -> None:
    """Run an ffmpeg command and surface stderr on failure."""
    ensure_ffmpeg_installed()

    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_output = completed.stderr.strip() or "ffmpeg exited with a non-zero status"
        raise RuntimeError(error_output)


def extract_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> Path:
    """Convert any supported input media file into a mono wav for diarization."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg_command(
        [
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(output_path),
        ]
    )
    return output_path


def cut_audio_clip(
    source_audio_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> Path:
    """Cut a diarized speaker segment into its own wav clip."""
    if end <= start:
        raise ValueError("Segment end time must be greater than the start time.")

    source_audio_path = Path(source_audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg_command(
        [
            "-i",
            str(source_audio_path),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(output_path),
        ]
    )
    return output_path
