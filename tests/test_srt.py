from mlx_whisper_yt2srt.srt import format_srt_timestamp, write_srt


def test_format_srt_timestamp():
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(65.4321) == "00:01:05,432"
    assert format_srt_timestamp(3661.9994) == "01:01:01,999"


def test_write_srt(tmp_path):
    output = tmp_path / "sample.srt"

    write_srt(
        [
            {"start": 0.0, "end": 1.25, "text": " Hello "},
            {"start": 1.25, "end": 2.5, "text": "world"},
        ],
        output,
    )

    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,250\nHello\n\n2\n00:00:01,250 --> 00:00:02,500\nworld\n\n"
    )
