from pathlib import Path

from mlx_whisper_yt2srt import app
from mlx_whisper_yt2srt.app import ConversionOptions


def test_conversion_options_defaults():
    options = ConversionOptions(youtube_url="https://youtu.be/example")

    assert options.language == "auto"
    assert options.model_size == "turbo-4bit"
    assert options.audio_format == "mp3"
    assert options.workspace_dir == Path("whisper_workspace")


def test_convert_youtube_to_srt_passes_progress_callback(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.mp3"
    srt_path = tmp_path / "audio.srt"
    events = []

    def fake_download_youtube_audio(url, *, audio_format, workspace_dir, progress):
        assert url == "https://youtu.be/example"
        assert audio_format == "mp3"
        assert workspace_dir == Path("work")
        progress("download progress")
        return audio_path

    def fake_generate_srt(audio_file, *, model_size, language, progress):
        assert audio_file == audio_path
        assert model_size == "tiny"
        assert language == "ja"
        progress("transcription progress")
        return srt_path

    monkeypatch.setattr(app, "download_youtube_audio", fake_download_youtube_audio)
    monkeypatch.setattr(app, "generate_srt", fake_generate_srt)

    result = app.convert_youtube_to_srt(
        ConversionOptions(
            youtube_url="https://youtu.be/example",
            language="ja",
            model_size="tiny",
            workspace_dir=Path("work"),
        ),
        progress=events.append,
    )

    assert result == srt_path
    assert events == [
        "download progress",
        "Starting subtitle generation...",
        "transcription progress",
    ]
