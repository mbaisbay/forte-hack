"""End-vs-start scene change outside people/laptop boxes (no model). -> data/residual.json {video: value}"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import frames

ROOT = Path(__file__).resolve().parent.parent


def mask_boxes(shape, boxes, pad=0.03):
    m = np.ones(shape[:2], bool); H, W = shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b[1:5]
        m[int(max(0, y1 - pad) * H):int(min(1, y2 + pad) * H), int(max(0, x1 - pad) * W):int(min(1, x2 + pad) * W)] = False
    return m
