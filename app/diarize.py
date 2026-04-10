"""Runs pyannote"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub.errors import GatedRepoError
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def format_hms(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def load_waveform(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load the extracted mono PCM wav without relying on torchcodec."""
    audio_path = Path(audio_path)

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        raw_frames = wav_file.readframes(num_frames)

    if sample_width != 2:
        raise RuntimeError(
            f"Unsupported WAV sample width {sample_width} bytes in {audio_path}. "
            "Expected 16-bit PCM audio."
        )

    waveform = torch.frombuffer(bytearray(raw_frames), dtype=torch.int16)
    waveform = waveform.reshape(-1, num_channels).transpose(0, 1).to(torch.float32)
    waveform = waveform / 32768.0
    return waveform, sample_rate


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
    waveform, sample_rate = load_waveform(audio_path)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    with ProgressHook() as hook:
        output = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "uri": Path(audio_path).stem,
            },
            hook=hook,
            **kwargs,
        )

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
                "start": format_hms(float(turn.start)),
                "end": format_hms(float(turn.end)),
                "duration_seconds": float(turn.end - turn.start),
            }
        )

    return segments, output
