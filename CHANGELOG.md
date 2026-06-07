# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning while it remains useful for a small CLI package.

## [0.1.0] - 2026-06-07

### Added

- Package entry point: `yt2srt`.
- YouTube URL to SRT conversion powered by `yt-dlp` and MLX Whisper.
- `uv` project setup with a committed lock file.
- Ruff formatting and linting.
- Pytest tests for CLI behavior, YouTube extraction boundaries, transcription
  result validation, and SRT formatting.
- GitHub Actions CI for Python 3.11 and 3.12 on macOS.

### Changed

- Restricted accepted URLs to YouTube hosts.
- Moved core progress reporting behind a callback so CLI output stays outside
  library logic.
- Raised the minimum `yt-dlp` version to the release used for current
  verification.

### Security

- Added early `ffmpeg` availability checks.
- Added Legal, Privacy, and Security notes to the README.
