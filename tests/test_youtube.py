import pytest

from mlx_whisper_yt2srt.errors import Yt2SrtError
from mlx_whisper_yt2srt.youtube import _extract_video_id, download_youtube_audio


class FakeYoutubeDL:
    download_called = False

    def __init__(self, options):
        self.options = options
        assert options["noplaylist"] is True
        assert options["outtmpl"].endswith("youtube_%(id)s.%(ext)s")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def extract_info(self, url, download=False):
        assert url == "https://youtu.be/example"
        assert download is False
        return {"id": "example12345"}

    def download(self, urls):
        assert urls == ["https://youtu.be/example"]
        FakeYoutubeDL.download_called = True
        output = (
            self.options["outtmpl"]
            .replace("%(id)s", "example12345")
            .replace(
                "%(ext)s",
                "mp3",
            )
        )
        with open(output, "w", encoding="utf-8") as file:
            file.write("audio")


def test_extract_video_id_requires_metadata():
    with pytest.raises(Yt2SrtError, match="metadata"):
        _extract_video_id({})


def test_extract_video_id_rejects_playlist_metadata():
    with pytest.raises(Yt2SrtError, match="Playlist URLs are not supported"):
        _extract_video_id({"entries": [{"id": "one"}]})


def test_download_youtube_audio_reuses_existing_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing_file = workspace / "youtube_example12345.mp3"
    existing_file.write_text("audio", encoding="utf-8")
    FakeYoutubeDL.download_called = False

    result = download_youtube_audio(
        "https://youtu.be/example",
        workspace_dir=workspace,
        youtube_dl_cls=FakeYoutubeDL,
    )

    assert result == existing_file
    assert FakeYoutubeDL.download_called is False


def test_download_youtube_audio_downloads_expected_file(tmp_path):
    FakeYoutubeDL.download_called = False

    result = download_youtube_audio(
        "https://youtu.be/example",
        workspace_dir=tmp_path,
        youtube_dl_cls=FakeYoutubeDL,
    )

    assert result == tmp_path / "youtube_example12345.mp3"
    assert result.read_text(encoding="utf-8") == "audio"
    assert FakeYoutubeDL.download_called is True
