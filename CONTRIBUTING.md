# Contributing

Thank you for considering a contribution.

This project is intentionally scoped to YouTube URLs on macOS with Apple
silicon. Please open an issue before proposing broad platform support,
non-YouTube URL support, or large backend abstractions.

## Development Setup

```bash
uv sync --dev
```

## Checks

Run the same checks used by CI before opening a pull request:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv build
```

## Design Guidelines

- Keep the CLI small and predictable.
- Keep user-facing terminal output in the CLI layer.
- Keep core conversion code testable without network access.
- Add focused tests for new behavior and edge cases.
- Do not add support for bypassing access controls, DRM, or platform
  restrictions.

## Reporting Bugs

Please include:

- `yt2srt --version`
- macOS version and hardware
- Python version
- command used
- relevant error output
- whether `ffmpeg` is available on `PATH`
