# mlx_whisper_yt2srt

A simple utility app that converts a YouTube URL into an SRT subtitle file,
powered by MLX Whisper.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run yt2srt "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Optional arguments:

```bash
uv run yt2srt "https://www.youtube.com/watch?v=XXXXXXXXXXX" --language ja --model turbo-4bit --audio-format mp3
```

You can also run the script directly:

```bash
uv run python yt2srt.py
```

Generated audio and SRT files are written to `whisper_workspace/`.
