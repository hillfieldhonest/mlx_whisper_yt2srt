from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .app import ConversionOptions, convert_youtube_to_srt
from .config import AUDIO_FORMATS, DEFAULT_AUDIO_FORMAT, DEFAULT_MODEL_SIZE, MODEL_REPOS
from .errors import Yt2SrtError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="yt2srt",
        description="Convert a YouTube video to an SRT subtitle file using MLX Whisper.",
    )
    parser.add_argument(
        "youtube_url",
        nargs="?",
        help="YouTube video URL, including youtube.com, youtu.be, shorts, and embed URLs.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="auto",
        help="Whisper language code such as en, ja, or auto. Default: auto.",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=tuple(MODEL_REPOS),
        default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size to use. Default: {DEFAULT_MODEL_SIZE}.",
    )
    parser.add_argument(
        "--audio-format",
        "-a",
        choices=AUDIO_FORMATS,
        default=DEFAULT_AUDIO_FORMAT,
        help=f"Audio format to download. Default: {DEFAULT_AUDIO_FORMAT}.",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default="whisper_workspace",
        type=Path,
        help="Directory for downloaded audio and generated SRT files.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt for missing URL, language, and model options.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive or not args.youtube_url:
        args = _prompt_for_missing_options(args)

    options = ConversionOptions(
        youtube_url=args.youtube_url,
        language=args.language,
        model_size=args.model,
        audio_format=args.audio_format,
        workspace_dir=args.workspace,
    )

    print(f"Processing URL: {options.youtube_url}")
    print(f"Using language: {options.language}")
    print(f"Using model: {options.model_size}")
    print(f"Audio format: {options.audio_format}")
    print(f"Workspace: {options.workspace_dir}")

    try:
        srt_file = convert_youtube_to_srt(options, progress=print)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Yt2SrtError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done! The SRT file is saved at: {srt_file}")
    return 0


def _prompt_for_missing_options(args: argparse.Namespace) -> argparse.Namespace:
    print("YouTube to SRT Converter")
    print("========================")

    if not args.youtube_url:
        args.youtube_url = input("Enter YouTube video URL: ").strip()

    if args.language == "auto":
        lang_input = input("Language code (ja, en, auto / default: auto): ").strip()
        args.language = lang_input or "auto"

    if args.model == DEFAULT_MODEL_SIZE:
        print("Available model sizes:")
        for model_name in MODEL_REPOS:
            print(f"- {model_name}")
        model_input = input(f"Model size (default: {DEFAULT_MODEL_SIZE}): ").strip()
        args.model = model_input or DEFAULT_MODEL_SIZE

    return args
