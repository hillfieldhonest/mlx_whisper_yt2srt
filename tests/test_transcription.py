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
