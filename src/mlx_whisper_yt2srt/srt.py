from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""

    total_milliseconds = max(0, round(seconds * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    total_minutes, seconds_part = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"


def write_srt(segments: Iterable[Mapping[str, object]], srt_path: Path) -> None:
    """Write transcription segments to an SRT file."""

    with srt_path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            start_time = format_srt_timestamp(float(segment["start"]))
            end_time = format_srt_timestamp(float(segment["end"]))
            text = str(segment["text"]).strip()

            file.write(f"{index}\n")
            file.write(f"{start_time} --> {end_time}\n")
            file.write(f"{text}\n\n")
