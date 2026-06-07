from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_SIZE, MODEL_REPOS
from .errors import Yt2SrtError
from .progress import ProgressCallback, emit_progress
from .srt import write_srt


def generate_srt(
    audio_file: Path,
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str = "auto",
    progress: ProgressCallback | None = None,
) -> Path:
    """Transcribe an audio file with MLX Whisper and write an SRT file."""

    audio_file = audio_file.expanduser().resolve()
    if not audio_file.exists():
        raise Yt2SrtError(f"Audio file not found: {audio_file}")

    model_repo = MODEL_REPOS.get(model_size.lower())
    if model_repo is None:
        allowed = ", ".join(MODEL_REPOS)
        raise Yt2SrtError(f"Unknown model '{model_size}'. Choose one of: {allowed}")

    emit_progress(progress, f"Loading MLX Whisper model '{model_repo}'...")

    transcribe_kwargs: dict[str, Any] = {}
    if language.lower() != "auto":
        transcribe_kwargs["language"] = language

    try:
        import mlx_whisper
    except ImportError as exc:
        raise Yt2SrtError(
            "MLX Whisper could not be imported. Run 'uv sync' to install project dependencies.",
        ) from exc

    try:
        result = mlx_whisper.transcribe(
            str(audio_file),
            path_or_hf_repo=model_repo,
            **transcribe_kwargs,
        )
    except Exception as exc:
        raise Yt2SrtError(f"MLX Whisper transcription failed: {exc}") from exc

    segments = _extract_segments(result)
    srt_path = unique_srt_path(audio_file, model_size)
    try:
        write_srt(segments, srt_path)
    except Exception as exc:
        raise Yt2SrtError(f"SRT file generation failed: {exc}") from exc

    emit_progress(progress, f"Generated SRT file: {srt_path}")
    return srt_path


def _extract_segments(result: object) -> Sequence[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        raise Yt2SrtError("MLX Whisper returned an invalid transcription result.")

    segments = result.get("segments")
    if not isinstance(segments, list):
        raise Yt2SrtError("MLX Whisper returned no transcription segments.")

    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping):
            raise Yt2SrtError(f"MLX Whisper segment {index} is not an object.")
        missing_fields = {"start", "end", "text"} - set(segment)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise Yt2SrtError(f"MLX Whisper segment {index} is missing: {missing}")

    return segments


def unique_srt_path(audio_file: Path, model_size: str) -> Path:
    """Return a non-conflicting SRT path next to the source audio file."""

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
