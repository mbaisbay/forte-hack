"""Make a synthetic H.264 .mov (moving square on a gradient). No competition data involved.

Usage: python scripts/make_synthetic.py out.mov [seconds] [fps]
"""

import sys
from fractions import Fraction

import av
import numpy as np


def make(path: str, seconds: float = 12.0, fps: int = 30, w: int = 464, h: int = 848) -> None:
    with av.open(path, mode="w", format="mov") as out:
        stream = out.add_stream("libx264", rate=Fraction(fps, 1))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        n = int(seconds * fps)
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[..., 0] = np.linspace(30, 90, w, dtype=np.uint8)[None, :]
        base[..., 2] = np.linspace(60, 120, h, dtype=np.uint8)[:, None]
        for i in range(n):
            img = base.copy()
            x = int((w - 80) * (i / max(1, n - 1)))
            y = h // 2 - 40
            img[y : y + 80, x : x + 80] = (230, 200, 40)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


if __name__ == "__main__":
    make(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 12.0, int(sys.argv[3]) if len(sys.argv) > 3 else 30)
