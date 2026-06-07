from __future__ import annotations

from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, PostProcessingError

from .config import DEFAULT_AUDIO_FORMAT
from .errors import Yt2SrtError


def download_youtube_audio(
    url: str,
    *,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    workspace_dir: Path = Path("whisper_workspace"),
    youtube_dl_cls: type[YoutubeDL] = YoutubeDL,
) -> Path:
    if not url.strip():
        raise Yt2SrtError("YouTube URL is required.")

    workspace_dir = workspace_dir.expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio/best",
        "hls_use_mpegts": True,
        "outtmpl": str(workspace_dir / "youtube_%(id)s.%(ext)s"),
        "overwrites": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
            },
        ],
    }

    try:
        with youtube_dl_cls(options) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = _extract_video_id(info)
            expected_file = workspace_dir / f"youtube_{video_id}.{audio_format}"

            if expected_file.exists():
                print(f"Audio file already exists: {expected_file}")
                return expected_file

            print("Downloading audio...")
            ydl.download([url])
    except (DownloadError, PostProcessingError) as exc:
        raise Yt2SrtError(f"YouTube audio download failed: {exc}") from exc
    except Exception as exc:
        raise Yt2SrtError(f"Unexpected YouTube download failure: {exc}") from exc

    if not expected_file.exists():
        raise Yt2SrtError(f"Downloaded audio file was not found: {expected_file}")

    print(f"Download complete: {expected_file}")
    return expected_file


def _extract_video_id(info: dict[str, Any] | None) -> str:
    if not info:
        raise Yt2SrtError("yt-dlp did not return video metadata.")

    video_id = info.get("id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise Yt2SrtError("yt-dlp did not return a video id.")

    return video_id
