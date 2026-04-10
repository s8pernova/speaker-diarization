"""Writes JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    return value


def write_manifest(output_path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize a diarization run manifest to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    output_path.write_text(
        json.dumps(_to_json_ready(manifest_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
