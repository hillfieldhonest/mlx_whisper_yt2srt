# mlx_whisper_yt2srt

Fast, local YouTube-to-SRT subtitle generation for Apple Silicon Macs.

`mlx_whisper_yt2srt` downloads audio from a single YouTube video and generates an
SRT subtitle file locally with
[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
It is designed for creators, researchers, and developers who want a small,
reproducible CLI that uses an Apple Silicon-friendly local inference stack
instead of uploading media to a hosted transcription service.

This project is intentionally small, but it is structured like a maintainable
Python CLI: reproducible `uv` environment, package entry point, tests, linting,
and CI.

Only YouTube video URLs are accepted. Although `yt-dlp` supports many other
sites, this project keeps the supported input scope narrow so behavior,
documentation, and support expectations stay clear.

## Why This Tool

- Apple Silicon first: MLX is built for efficient machine learning on Apple
  silicon, and MLX Whisper lets this CLI use that stack through a simple Python
  API.
- Local-first workflow: audio and generated subtitles stay on your machine.
- Narrow scope: one YouTube video in, one SRT file out.
- Reproducible setup: `uv`, a committed lock file, CI, tests, and package entry
  points are included.
- Honest positioning: this project does not claim to be the fastest Whisper
  implementation. Performance claims should be backed by reproducible
  benchmarks because results vary by hardware, model, quantization, language,
  timestamp settings, and cold versus warm model caches.

## Scope

This is a local-first CLI for macOS on Apple silicon. It is intentionally not a
general-purpose `yt-dlp` frontend, hosted captioning service, or downloader for
non-YouTube platforms.

## Requirements

- macOS on Apple silicon for MLX / Metal acceleration
- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` available on `PATH`

On macOS, `ffmpeg` can be installed with Homebrew:

```bash
brew install ffmpeg
```

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

You can also run the module entry point:

```bash
uv run python -m mlx_whisper_yt2srt "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Generated audio and SRT files are written to `whisper_workspace/`.

Show the installed version:

```bash
uv run yt2srt --version
```

Example terminal output:

```text
Processing URL: https://www.youtube.com/watch?v=XXXXXXXXXXX
Using language: auto
Using model: turbo-4bit
Audio format: mp3
Workspace: whisper_workspace
Downloading audio...
Download complete: /path/to/whisper_workspace/youtube_XXXXXXXXXXX.mp3
Starting subtitle generation...
Loading MLX Whisper model 'mlx-community/whisper-large-v3-turbo-q4'...
Generated SRT file: /path/to/whisper_workspace/youtube_XXXXXXXXXXX_turbo-4bit.srt
Done! The SRT file is saved at: /path/to/whisper_workspace/youtube_XXXXXXXXXXX_turbo-4bit.srt
```

Example SRT output:

```srt
1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
This is an example subtitle.
```

## Options

```text
--language, -l      Whisper language code, such as auto, ja, or en.
--model, -m         tiny, base, small, medium, large, large-4bit, turbo, turbo-4bit.
--audio-format, -a  mp3, wav, or m4a.
--workspace, -w     Directory for downloaded audio and generated SRT files.
--interactive, -i   Prompt for missing options.
--version           Show the installed version.
```

## Development

```bash
uv sync --dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Legal and Acceptable Use

Use this tool only with content that you own, are authorized to process, or may
lawfully use. This project does not encourage bypassing access controls,
violating platform terms, or redistributing copyrighted audio, video, or
generated subtitles without permission.

You are responsible for complying with YouTube's terms, copyright law, and any
local rules that apply to the content you process. This README is not legal
advice.

## Privacy and Local Files

Downloaded audio and generated SRT files are stored locally in the workspace
directory, which defaults to `whisper_workspace/`. The workspace is ignored by
git, but the files may still contain sensitive speech, names, or other personal
data. Review generated files before sharing them.

First runs may download MLX Whisper model weights from Hugging Face. The exact
download size depends on the selected model and may require hundreds of
megabytes of disk space.

## Security Notes

- Non-YouTube URLs are rejected before extraction.
- `yt-dlp` is used through its Python API instead of shelling out.
- `ffmpeg` is checked before download and must be installed from a trusted
  source.
- Model weights are loaded through `mlx-whisper`; review the configured model
  repositories in `src/mlx_whisper_yt2srt/config.py` if supply-chain control is
  important for your environment.

## Design Notes

- The process working directory is never changed during conversion.
- Playlist URLs are rejected by default; pass a single YouTube video URL.
- MLX Whisper is imported lazily so `yt2srt --help` works even when Metal is not
  available in the current execution context.
- Core conversion code reports progress through a callback; the CLI owns
  user-facing terminal output.
- Model names are defined once in `src/mlx_whisper_yt2srt/config.py` and reused
  by the CLI.

## Performance Strategy

MLX Whisper is the default backend because it fits this project's current
product shape: Python packaging, Apple Silicon, local execution, Hugging Face
model distribution, and a small dependency surface. It is the right default for
a focused Python CLI even though it should not be marketed as universally
fastest without a benchmark.

The project should track the Apple Silicon speech-to-text ecosystem, especially
[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), and
[WhisperKit / Argmax OSS](https://github.com/argmaxinc/argmax-oss-swift).
Backend expansion should be driven by measured results rather than speculation.

A backend change or new backend option should require:

- a reproducible benchmark on representative Apple Silicon hardware;
- comparable model size, language, accuracy, and timestamp requirements;
- a clear install and maintenance story for end users;
- no regression to the one-command YouTube-to-SRT workflow.

Until those conditions are met, the tool stays focused on MLX Whisper.

## Limitations

- First runs may download Whisper model weights from Hugging Face.
- YouTube extraction can fail for unavailable, region-limited, or protected
  videos.
- MLX transcription requires a usable Metal device. Headless or sandboxed
  environments may need elevated host permissions to expose the GPU.
- Long videos can take a long time to download and transcribe.
