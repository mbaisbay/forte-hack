"""Safe door state on grounded safe crop (VLM multiple choice), every frame. -> data/safe_<tag>/<video>.json {t: A|B|C}"""
import argparse, json, re, sys
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel
sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from vlm import backends

ROOT = Path(__file__).resolve().parent.parent
Q = """This is a zoomed crop around a grey metal safe in a bank cashier room.
Look at the FRONT DOOR of the safe. Choose one:
A) the safe door is open (swung outward, or the dark inside of the safe is visible)
B) the safe door is closed (flat front, flush with the body)
C) the safe is hidden / not visible
Answer with one letter only."""


def crop(im, g):
    W, H = im.size; pw = (g[2] - g[0]) * 0.35; ph = (g[3] - g[1]) * 0.2
    c = im.crop((int(max(0, g[0] - pw) * W), int(max(0, g[1] - ph) * H), int(min(1, g[2] + pw * 0.3) * W), int(min(1, g[3] + ph) * H)))
    s = 448 / max(c.size)
    return c.resize((int(c.width * s), int(c.height * s)))
