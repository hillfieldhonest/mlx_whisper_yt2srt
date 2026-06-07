from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .progress import ProgressCallback, emit_progress
from .transcription import generate_srt
from .youtube import download_youtube_audio


@dataclass(frozen=True)
class ConversionOptions:
    """Options for one YouTube-to-SRT conversion."""

    youtube_url: str
    language: str = "auto"
    model_size: str = "turbo-4bit"
    audio_format: str = "mp3"
    workspace_dir: Path = Path("whisper_workspace")


def convert_youtube_to_srt(
    options: ConversionOptions,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download YouTube audio, transcribe it, and return the generated SRT path."""

    audio_path = download_youtube_audio(
        options.youtube_url,
        audio_format=options.audio_format,
        workspace_dir=options.workspace_dir,
        progress=progress,
    )
    emit_progress(progress, "Starting subtitle generation...")
    return generate_srt(
        audio_path,
        model_size=options.model_size,
        language=options.language,
        progress=progress,
    )
