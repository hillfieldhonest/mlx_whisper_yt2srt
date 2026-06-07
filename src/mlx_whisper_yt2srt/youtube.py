from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, PostProcessingError

from .config import DEFAULT_AUDIO_FORMAT
from .errors import Yt2SrtError
from .progress import ProgressCallback, emit_progress

YOUTUBE_DOMAINS = ("youtube.com", "youtube-nocookie.com")
YOUTUBE_SHORT_DOMAIN = "youtu.be"


def download_youtube_audio(
    url: str,
    *,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    workspace_dir: Path = Path("whisper_workspace"),
    youtube_dl_cls: type[YoutubeDL] = YoutubeDL,
    ffmpeg_checker: Callable[[str], str | None] = shutil.which,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download and extract audio from a single YouTube URL."""

    if not url.strip():
        raise Yt2SrtError("YouTube URL is required.")
    _ensure_youtube_url(url)
    _ensure_ffmpeg_available(ffmpeg_checker)

    workspace_dir = workspace_dir.expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio/best",
        "hls_use_mpegts": True,
        "noplaylist": True,
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
                emit_progress(progress, f"Audio file already exists: {expected_file}")
                return expected_file

            emit_progress(progress, "Downloading audio...")
            ydl.download([url])
    except (DownloadError, PostProcessingError) as exc:
        raise Yt2SrtError(f"YouTube audio download failed: {exc}") from exc
    except Exception as exc:
        raise Yt2SrtError(f"Unexpected YouTube download failure: {exc}") from exc

    if not expected_file.exists():
        raise Yt2SrtError(f"Downloaded audio file was not found: {expected_file}")

    emit_progress(progress, f"Download complete: {expected_file}")
    return expected_file


def _ensure_ffmpeg_available(
    ffmpeg_checker: Callable[[str], str | None] = shutil.which,
) -> None:
    if ffmpeg_checker("ffmpeg") is None:
        raise Yt2SrtError(
            "ffmpeg was not found on PATH. Install ffmpeg and ensure it is available on PATH.",
        )


def _ensure_youtube_url(url: str) -> None:
    parsed = urlparse(url.strip())
    hostname = parsed.hostname
    if not hostname:
        raise Yt2SrtError("Only YouTube URLs are supported; pass a youtube.com or youtu.be URL.")

    hostname = hostname.rstrip(".").lower()
    if hostname == YOUTUBE_SHORT_DOMAIN:
        return

    if any(hostname == domain or hostname.endswith(f".{domain}") for domain in YOUTUBE_DOMAINS):
        return

    raise Yt2SrtError("Only YouTube URLs are supported; pass a youtube.com or youtu.be URL.")


def _extract_video_id(info: dict[str, Any] | None) -> str:
    if not info:
        raise Yt2SrtError("yt-dlp did not return video metadata.")

    if "entries" in info:
        raise Yt2SrtError("Playlist URLs are not supported; pass a single video URL.")

    video_id = info.get("id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise Yt2SrtError("yt-dlp did not return a video id.")

    return video_id
