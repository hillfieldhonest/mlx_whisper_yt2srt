from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_SIZE, MODEL_REPOS
from .errors import Yt2SrtError
from .srt import write_srt


def generate_srt(
    audio_file: Path,
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str = "auto",
) -> Path:
    audio_file = audio_file.expanduser().resolve()
    if not audio_file.exists():
        raise Yt2SrtError(f"Audio file not found: {audio_file}")

    model_repo = MODEL_REPOS.get(model_size.lower())
    if model_repo is None:
        allowed = ", ".join(MODEL_REPOS)
        raise Yt2SrtError(f"Unknown model '{model_size}'. Choose one of: {allowed}")

    print(f"Loading MLX Whisper model '{model_repo}'...")

    transcribe_kwargs: dict[str, Any] = {}
    if language.lower() != "auto":
        transcribe_kwargs["language"] = language

    try:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(audio_file),
            path_or_hf_repo=model_repo,
            **transcribe_kwargs,
        )
    except Exception as exc:
        raise Yt2SrtError(f"MLX Whisper transcription failed: {exc}") from exc

    srt_path = unique_srt_path(audio_file, model_size)
    write_srt(result["segments"], srt_path)
    print(f"Generated SRT file: {srt_path}")
    return srt_path


def unique_srt_path(audio_file: Path, model_size: str) -> Path:
    base_path = audio_file.with_name(f"{audio_file.stem}_{model_size.lower()}.srt")
    if not base_path.exists():
        return base_path

    counter = 2
    while True:
        candidate = audio_file.with_name(
            f"{audio_file.stem}_{model_size.lower()}_{counter:02d}.srt",
        )
        if not candidate.exists():
            return candidate
        counter += 1
