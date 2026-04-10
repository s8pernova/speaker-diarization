"""Merges clips per speaker."""

from __future__ import annotations

import shutil
from pathlib import Path

try:
    from .ffmpeg_utils import run_ffmpeg_command
except ImportError:
    from ffmpeg_utils import run_ffmpeg_command


def _concat_entry(path: Path) -> str:
    resolved = str(path.resolve()).replace("'", r"'\''")
    return f"file '{resolved}'\n"


def merge_speaker_clips(clip_paths: list[str | Path], output_path: str | Path) -> Path:
    """Merge a speaker's ordered clips into a single wav pack."""
    normalized_clip_paths = [Path(path) for path in clip_paths]
    if not normalized_clip_paths:
        raise ValueError("At least one clip is required to create a merged speaker pack.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(normalized_clip_paths) == 1:
        shutil.copyfile(normalized_clip_paths[0], output_path)
        return output_path

    concat_file = output_path.parent / f".{output_path.stem}.concat.txt"
    try:
        with concat_file.open("w", encoding="utf-8") as handle:
            for clip_path in normalized_clip_paths:
                handle.write(_concat_entry(clip_path))

        run_ffmpeg_command(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]
        )
    finally:
        concat_file.unlink(missing_ok=True)

    return output_path
