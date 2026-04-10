"""Groups timestamps by speaker."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def normalize_speaker_key(speaker: str) -> str:
    """Create a filesystem-safe speaker key."""
    speaker_key = re.sub(r"[^0-9A-Za-z]+", "_", speaker).strip("_").lower()
    return speaker_key or "speaker"


def group_segments_by_speaker(
    raw_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Sort diarization results and group them under normalized speaker labels."""
    ordered_segments: list[dict[str, Any]] = []
    grouped_segments: dict[str, dict[str, Any]] = {}
    speaker_clip_indexes: defaultdict[str, int] = defaultdict(int)

    sorted_segments = sorted(
        raw_segments,
        key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])),
    )

    for raw_segment in sorted_segments:
        start = float(raw_segment["start"])
        end = float(raw_segment["end"])
        if end <= start:
            continue

        speaker = str(raw_segment["speaker"])
        speaker_key = normalize_speaker_key(speaker)
        speaker_clip_indexes[speaker_key] += 1

        segment = {
            "index": len(ordered_segments) + 1,
            "speaker": speaker,
            "speaker_key": speaker_key,
            "speaker_clip_index": speaker_clip_indexes[speaker_key],
            "start": start,
            "end": end,
            "duration": float(raw_segment.get("duration", end - start)),
        }
        ordered_segments.append(segment)

        speaker_group = grouped_segments.setdefault(
            speaker_key,
            {
                "speaker": speaker,
                "speaker_key": speaker_key,
                "segment_count": 0,
                "total_duration": 0.0,
                "segments": [],
            },
        )
        speaker_group["segments"].append(segment)
        speaker_group["segment_count"] += 1
        speaker_group["total_duration"] += segment["duration"]

    return ordered_segments, grouped_segments
