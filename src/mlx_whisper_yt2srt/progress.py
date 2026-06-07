from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[str], None]


def emit_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
