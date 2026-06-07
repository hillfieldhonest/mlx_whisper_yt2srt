import pytest

from mlx_whisper_yt2srt.errors import Yt2SrtError
from mlx_whisper_yt2srt.youtube import (
    _ensure_ffmpeg_available,
    _ensure_youtube_url,
    _extract_video_id,
    download_youtube_audio,
)


def ffmpeg_found(_binary):
    return "/usr/local/bin/ffmpeg"


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


@pytest.fixture(autouse=True)
def reset_fake_youtube_dl():
    FakeYoutubeDL.download_called = False


def test_extract_video_id_requires_metadata():
    with pytest.raises(Yt2SrtError, match="metadata"):
        _extract_video_id({})


def test_extract_video_id_requires_metadata_when_none():
    with pytest.raises(Yt2SrtError, match="metadata"):
        _extract_video_id(None)


def test_extract_video_id_rejects_playlist_metadata():
    with pytest.raises(Yt2SrtError, match="Playlist URLs are not supported"):
        _extract_video_id({"entries": [{"id": "one"}]})


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=example12345",
        "https://m.youtube.com/watch?v=example12345",
        "https://music.youtube.com/watch?v=example12345",
        "https://youtube.com/shorts/example12345",
        "https://youtu.be/example12345",
        "https://www.youtube-nocookie.com/embed/example12345",
    ],
)
def test_ensure_youtube_url_accepts_youtube_hosts(url):
    _ensure_youtube_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/12345",
        "https://notyoutube.com/watch?v=example12345",
        "https://youtube.com.example.org/watch?v=example12345",
        "example12345",
        "",
        "   ",
    ],
)
def test_ensure_youtube_url_rejects_non_youtube_hosts(url):
    with pytest.raises(Yt2SrtError, match="Only YouTube URLs are supported"):
        _ensure_youtube_url(url)


def test_ensure_ffmpeg_available_requires_ffmpeg():
    with pytest.raises(Yt2SrtError, match="ffmpeg was not found"):
        _ensure_ffmpeg_available(lambda _binary: None)


def test_download_youtube_audio_reuses_existing_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing_file = workspace / "youtube_example12345.mp3"
    existing_file.write_text("audio", encoding="utf-8")

    result = download_youtube_audio(
        "https://youtu.be/example",
        workspace_dir=workspace,
        youtube_dl_cls=FakeYoutubeDL,
        ffmpeg_checker=ffmpeg_found,
    )

    assert result == existing_file
    assert FakeYoutubeDL.download_called is False


def test_download_youtube_audio_downloads_expected_file(tmp_path):
    result = download_youtube_audio(
        "https://youtu.be/example",
        workspace_dir=tmp_path,
        youtube_dl_cls=FakeYoutubeDL,
        ffmpeg_checker=ffmpeg_found,
    )

    assert result == tmp_path / "youtube_example12345.mp3"
    assert result.read_text(encoding="utf-8") == "audio"
    assert FakeYoutubeDL.download_called is True


def test_download_youtube_audio_requires_ffmpeg_before_extraction(tmp_path):
    with pytest.raises(Yt2SrtError, match="ffmpeg was not found"):
        download_youtube_audio(
            "https://youtu.be/example",
            workspace_dir=tmp_path,
            youtube_dl_cls=FakeYoutubeDL,
            ffmpeg_checker=lambda _binary: None,
        )

    assert FakeYoutubeDL.download_called is False


def test_download_youtube_audio_reports_progress_via_callback(tmp_path, capsys):
    events = []

    result = download_youtube_audio(
        "https://youtu.be/example",
        workspace_dir=tmp_path,
        youtube_dl_cls=FakeYoutubeDL,
        ffmpeg_checker=ffmpeg_found,
        progress=events.append,
    )

    captured = capsys.readouterr()
    assert result == tmp_path / "youtube_example12345.mp3"
    assert events == [
        "Downloading audio...",
        f"Download complete: {tmp_path / 'youtube_example12345.mp3'}",
    ]
    assert captured.out == ""
