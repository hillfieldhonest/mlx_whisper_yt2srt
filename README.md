# mlx_whisper_yt2srt

Convert a YouTube video URL into an SRT subtitle file using
[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
and `yt-dlp`.

This project is intentionally small, but it is structured like a maintainable
Python CLI: reproducible `uv` environment, package entry point, tests, linting,
and CI.

## Requirements

- macOS on Apple silicon for MLX / Metal acceleration
- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` available on `PATH`

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

## Options

```text
--language, -l      Whisper language code, such as auto, ja, or en.
--model, -m         tiny, base, small, medium, large, large-4bit, turbo, turbo-4bit.
--audio-format, -a  mp3, wav, or m4a.
--workspace, -w     Directory for downloaded audio and generated SRT files.
--interactive, -i   Prompt for missing options.
```

## Development

```bash
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Design Notes

- `yt-dlp` is used through its Python API instead of shelling out.
- The process working directory is never changed during conversion.
- MLX Whisper is imported lazily so `yt2srt --help` works even when Metal is not
  available in the current execution context.
- Model names are defined once in `src/mlx_whisper_yt2srt/config.py` and reused
  by the CLI.

## Limitations

- First runs may download Whisper model weights from Hugging Face.
- YouTube extraction can fail for unavailable, region-limited, or protected
  videos.
- MLX transcription requires a usable Metal device. Headless or sandboxed
  environments may need elevated host permissions to expose the GPU.
