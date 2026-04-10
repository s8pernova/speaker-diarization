"""Runs pyannote"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from huggingface_hub.errors import GatedRepoError
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def build_pipeline(hf_token: str, use_gpu: bool = True) -> Pipeline:
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Cannot access `pyannote/speaker-diarization-community-1`. "
            "Accept the model access request on Hugging Face and use a token "
            "that has permission for that repo."
        ) from exc

    if use_gpu and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    return pipeline


def run_diarization(
    audio_path: str | Path,
    hf_token: str,
    use_gpu: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    use_exclusive: bool = True,
):
    pipeline = build_pipeline(hf_token=hf_token, use_gpu=use_gpu)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    with ProgressHook() as hook:
        output = pipeline(str(audio_path), hook=hook, **kwargs)

    diarization = (
        output.exclusive_speaker_diarization
        if use_exclusive and hasattr(output, "exclusive_speaker_diarization")
        else output.speaker_diarization
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
                "duration": float(turn.end - turn.start),
            }
        )

    return segments, output
