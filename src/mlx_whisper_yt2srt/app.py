from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .transcription import generate_srt
from .youtube import download_youtube_audio


@dataclass(frozen=True)
class ConversionOptions:
    youtube_url: str
    language: str = "auto"
    model_size: str = "turbo-4bit"
    audio_format: str = "mp3"
    workspace_dir: Path = Path("whisper_workspace")


def convert_youtube_to_srt(options: ConversionOptions) -> Path:
    audio_path = download_youtube_audio(
        options.youtube_url,
        audio_format=options.audio_format,
        workspace_dir=options.workspace_dir,
    )
    print("Starting subtitle generation...")
    return generate_srt(
        audio_path,
        model_size=options.model_size,
        language=options.language,
    )
