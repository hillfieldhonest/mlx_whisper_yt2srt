from __future__ import annotations

DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_MODEL_SIZE = "turbo-4bit"

AUDIO_FORMATS = ("mp3", "wav", "m4a")

MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-4bit": "mlx-community/whisper-large-v3-mlx-4bit",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo-4bit": "mlx-community/whisper-large-v3-turbo-q4",
}
