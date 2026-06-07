from pathlib import Path

from mlx_whisper_yt2srt import cli
from mlx_whisper_yt2srt.errors import Yt2SrtError


def test_main_returns_zero_when_conversion_succeeds(monkeypatch, tmp_path, capsys):
    output_file = tmp_path / "out.srt"

    def fake_convert(options):
        assert options.youtube_url == "https://youtu.be/example"
        assert options.language == "ja"
        assert options.model_size == "tiny"
        assert options.audio_format == "mp3"
        assert options.workspace_dir == Path("work")
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
    assert f"Done! The SRT file is saved at: {output_file}" in captured.out


def test_main_returns_one_when_conversion_fails(monkeypatch, capsys):
    def fake_convert(_options):
        raise Yt2SrtError("download failed")

    monkeypatch.setattr(cli, "convert_youtube_to_srt", fake_convert)

    exit_code = cli.main(["https://youtu.be/example"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: download failed" in captured.err
