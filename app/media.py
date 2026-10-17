"""Video probing, decoding with real timestamps, and JPEG frame handling."""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import av
from PIL import Image


class MediaError(Exception):
    pass


@dataclass
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    rotation: int
    codec: str


@dataclass
class Frame:
    frame_id: str           # "F0042": what the model sees and refers back to
    timestamp_sec: float    # seconds from the first displayed frame (decoder PTS)
    model_jpeg: bytes       # downscaled, sent to the VLM
    evidence_jpeg: bytes    # original resolution, returned to the jury


@dataclass
class DecodedVideo:
    info: VideoInfo
    frames: list[Frame] = field(default_factory=list)   # sampled at verify_fps

    def by_id(self, frame_id: str) -> Frame | None:
        return self._index.get(frame_id)

    def __post_init__(self):
        self._index = {f.frame_id: f for f in self.frames}

    def between(self, start: float, end: float) -> list[Frame]:
        return [f for f in self.frames if start <= f.timestamp_sec <= end]


def _rotation(frame) -> int:
    return int(round(getattr(frame, "rotation", 0) or 0)) % 360


def probe(path: Path) -> VideoInfo:
    """Open the file and decode one frame. Raises MediaError if it is not a usable video."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise MediaError("no video stream")
            stream = container.streams.video[0]
            first = next(container.decode(stream), None)
            if first is None:
                raise MediaError("no decodable video frame")
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / 1_000_000
            else:
                raise MediaError("unknown duration")
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            rot = _rotation(first)
            w, h = first.width, first.height
            if rot in (90, 270):
                w, h = h, w
            return VideoInfo(duration, fps, w, h, rot, stream.codec_context.name)
    except MediaError:
        raise
    except Exception as exc:  # PyAV raises many error types for broken media
        raise MediaError(f"cannot decode video: {type(exc).__name__}") from exc


def _to_image(frame) -> Image.Image:
    img = frame.to_image()
    rot = _rotation(frame)
    if rot:
        # PyAV reports the counter-clockwise display rotation; PIL rotates counter-clockwise too.
        img = img.rotate(rot, expand=True)
    return img


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)


def decode(path: Path, sample_fps: float, model_max_side: int) -> DecodedVideo:
    """Decode the whole video once, keeping frames on a sample_fps grid.

    Timestamps come from decoder PTS relative to the first frame, not index / nominal fps.
    """
    info = probe(path)
    step = 1.0 / sample_fps
    frames: list[Frame] = []
    next_t = 0.0
    first_pts_sec = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            pts_sec = float(frame.pts * frame.time_base)
            if first_pts_sec is None:
                first_pts_sec = pts_sec
            t = pts_sec - first_pts_sec
            if t + 1e-6 < next_t:
                continue
            img = _to_image(frame)
            frames.append(
                Frame(
                    frame_id=f"F{len(frames):04d}",
                    timestamp_sec=round(t, 3),
                    model_jpeg=encode_jpeg(downscale(img, model_max_side), 85),
                    evidence_jpeg=encode_jpeg(img, 90),
                )
            )
            # Skip ahead on the grid if frames are sparse, so timestamps never drift backwards.
            while next_t <= t + 1e-6:
                next_t += step
    if not frames:
        raise MediaError("no frames decoded")
    return DecodedVideo(info=info, frames=frames)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
