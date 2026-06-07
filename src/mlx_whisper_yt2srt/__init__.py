"""YouTube to SRT conversion powered by MLX Whisper."""

from importlib.metadata import PackageNotFoundError, version

from .app import ConversionOptions, convert_youtube_to_srt

try:
    __version__ = version("mlx-whisper-yt2srt")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["ConversionOptions", "__version__", "convert_youtube_to_srt"]
