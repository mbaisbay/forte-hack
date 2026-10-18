"""Frame sampling with burned-in timestamps."""
import cv2
import numpy as np
from PIL import Image
import hashlib, os, shutil, tempfile


def _open(path):
    """cv2 on Windows can't open non-ASCII paths (Х-*.mov) -> copy to ASCII temp name."""
    path = str(path)
    if not path.isascii():
        tmp = os.path.join(tempfile.gettempdir(), "cashvid_" + hashlib.md5(path.encode()).hexdigest() + ".mov")
        if not os.path.exists(tmp) or os.path.getsize(tmp) != os.path.getsize(path):
            shutil.copyfile(path, tmp)
        path = tmp
    return cv2.VideoCapture(path)


def sample(path, fps=2.0, max_frames=64, width=320, overlay=True):
    """Return [(t_sec, PIL.Image)] sampled uniformly; fps lowered if video is long."""
    cap = _open(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / vfps
    step = max(1.0 / fps, dur / max_frames)
    out, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * vfps))
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        if overlay:
            txt = f"t={t:.1f}s"
            cv2.rectangle(f, (0, 0), (110, 26), (0, 0, 0), -1)
            cv2.putText(f, txt, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.append((round(t, 1), Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))
        t += step
    cap.release()
    return out


def frame_at(path, t, width=464):
    """Full-res JPEG bytes of frame at t (for evidence)."""
    cap = _open(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def contact_sheet(frames, cols=8):
    ims = [np.asarray(im) for _, im in frames]
    h, w = ims[0].shape[:2]
    rows = (len(ims) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return Image.fromarray(sheet)
