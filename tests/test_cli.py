import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from mlx_whisper_yt2srt import __version__, cli
from mlx_whisper_yt2srt.errors import Yt2SrtError


def test_main_returns_zero_when_conversion_succeeds(monkeypatch, tmp_path, capsys):
    output_file = tmp_path / "out.srt"

    def fake_convert(options, *, progress):
        assert options.youtube_url == "https://youtu.be/example"
        assert options.language == "ja"
        assert options.model_size == "tiny"
        assert options.audio_format == "mp3"
        assert options.workspace_dir == Path("work")
        progress("core progress")
        return output_file

    monkeypatch.setattr(cli, "convert_youtube_to_srt", fake_convert)

    exit_code = cli.main(
        [
            "https://youtu.be/example",
            "--language",
            "ja",
            "--model",
            "tiny",
            "--workspace",
            "work",
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "core progress" in captured.out
    assert f"Done! The SRT file is saved at: {output_file}" in captured.out


def test_main_returns_one_when_conversion_fails(monkeypatch, capsys):
    def fake_convert(_options, *, progress):
        assert callable(progress)
        raise Yt2SrtError("download failed")

    monkeypatch.setattr(cli, "convert_youtube_to_srt", fake_convert)

    exit_code = cli.main(["https://youtu.be/example"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: download failed" in captured.err


def test_module_entrypoint_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_whisper_yt2srt", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Convert a YouTube video to an SRT subtitle file" in result.stdout


def test_package_version_comes_from_installed_metadata():
    assert __version__ == version("mlx-whisper-yt2srt")


def test_module_entrypoint_version_runs():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_whisper_yt2srt", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"yt2srt {__version__}"
