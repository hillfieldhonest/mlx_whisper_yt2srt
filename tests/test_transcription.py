import sys
from types import SimpleNamespace

import pytest

from mlx_whisper_yt2srt import transcription
from mlx_whisper_yt2srt.errors import Yt2SrtError
from mlx_whisper_yt2srt.transcription import unique_srt_path


def test_unique_srt_path_returns_base_path_when_available(tmp_path):
    audio_file = tmp_path / "youtube_abc.mp3"
    audio_file.touch()

    assert unique_srt_path(audio_file, "tiny") == tmp_path / "youtube_abc_tiny.srt"


def test_unique_srt_path_adds_counter_when_file_exists(tmp_path):
    audio_file = tmp_path / "youtube_abc.mp3"
    audio_file.touch()
    (tmp_path / "youtube_abc_tiny.srt").touch()
    (tmp_path / "youtube_abc_tiny_02.srt").touch()

    assert unique_srt_path(audio_file, "tiny") == tmp_path / "youtube_abc_tiny_03.srt"


def test_generate_srt_rejects_transcription_without_segments(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_text("audio", encoding="utf-8")
    fake_mlx = SimpleNamespace(transcribe=lambda *_args, **_kwargs: {})
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

    with pytest.raises(Yt2SrtError, match="no transcription segments"):
        transcription.generate_srt(audio_file, model_size="tiny")


def test_generate_srt_wraps_srt_write_failure(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.mp3"
    audio_file.write_text("audio", encoding="utf-8")
    fake_mlx = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: {
            "segments": [{"start": 0, "end": 1, "text": "hello"}],
        },
    )
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_mlx)

    def fail_write_srt(_segments, _srt_path):
        raise OSError("disk full")

    monkeypatch.setattr(transcription, "write_srt", fail_write_srt)

    with pytest.raises(Yt2SrtError, match="SRT file generation failed"):
        transcription.generate_srt(audio_file, model_size="tiny")
