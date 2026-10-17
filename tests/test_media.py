import pytest

from app.media import MediaError, decode, probe


def test_probe_synthetic(synthetic_video):
    info = probe(synthetic_video)
    assert info.width == 464 and info.height == 848
    assert 11.5 < info.duration_sec < 12.5
    assert info.rotation == 0


def test_probe_rejects_corrupt(corrupt_file):
    with pytest.raises(MediaError):
        probe(corrupt_file)


def test_decode_grid_and_timestamps(synthetic_video):
    video = decode(synthetic_video, sample_fps=6, model_max_side=640)
    ts = [f.timestamp_sec for f in video.frames]
    assert ts[0] == 0.0
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert 68 <= len(ts) <= 74  # ~12 s at 6 fps
    assert ts[-1] <= video.info.duration_sec
    f = video.frames[10]
    assert f.evidence_jpeg[:2] == b"\xff\xd8" and f.model_jpeg[:2] == b"\xff\xd8"
    assert video.by_id(f.frame_id) is f
