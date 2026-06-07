# Security Policy

## Supported Versions

Security fixes are handled on the `main` branch while the project is in the
`0.x` series.

## Reporting a Vulnerability

If you find a vulnerability, please open a private security advisory on GitHub
or contact the maintainer through the repository owner account. Do not publish
exploitable details until there has been a reasonable opportunity to assess and
fix the issue.

Please include:

- affected version or commit
- operating system and Python version
- steps to reproduce
- whether the issue involves a malicious URL, generated file, dependency, or
  model download

## Security Scope

This project accepts only YouTube URLs and delegates extraction to `yt-dlp`.
`ffmpeg` must be installed separately and should come from a trusted source.
MLX Whisper model weights may be downloaded from Hugging Face on first use.

This project does not support bypassing access controls, DRM, or platform
restrictions.
