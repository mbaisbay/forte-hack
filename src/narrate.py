"""Rolling narration: every WIN seconds, VLM sees the window frames + previous state line, writes an updated state line.
Output: data/narr_<tag>/<video>.txt (one line per window).
  python src/narrate.py --model models/gemma-3-4b-it --tag gemma4b [videos...]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frames, vlm
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
SYS = ("You describe overhead CCTV of a small bank cashier room: desk with laptop and banknote counting machine, "
       "grey metal safe, glass door. The CASHIER works behind the desk (laptop/safe side). Clients or couriers enter "
       "through the door. Report only what is visible; never guess.")
PROMPT = """Previous state (may be wrong - trust the frames if they disagree):
{PREV}

Frames above cover t={A:.0f}-{B:.0f}s. Write ONE updated line in exactly this format:
t={A:.0f}-{B:.0f}s | cashier: <at desk / standing / away from desk / left room> | others: <none / client seated / client standing / courier / person behind desk> | door: <open/closed> | safe: <open/closed> | laptop screen: <app or table / lock screen / off / not visible> | desk: <list visible items: banknotes, ID card, paper, bag, cup, phone, personal items, nothing> | action: <what happens in these seconds, who hands what to whom>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--model", default="models/gemma-3-4b-it")
    ap.add_argument("--quant", default="none")
    ap.add_argument("--tag", default="gemma4b")
    ap.add_argument("--win", type=float, default=3)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--width", type=int, default=448)
    a = ap.parse_args()
    gt = json.loads((ROOT / "data/gt.json").read_text(encoding="utf-8"))
    d = ROOT / f"data/narr_{a.tag}"; d.mkdir(parents=True, exist_ok=True)
    for v in (a.videos or gt):
        f = d / f"{v}.txt"
        if f.exists():
            continue
        t0 = time.time()
        fr = frames.sample(video_path(v), fps=a.fps, max_frames=10_000, width=a.width, overlay=True)
        dur = fr[-1][0]
        prev, lines, s = "(start of video)", [], 0.0
        while s <= dur:
            w = [x for x in fr if s <= x[0] < s + a.win]
            if w:
                c = []
                for t, im in w:
                    c += [f"t={t:.1f}s", im]
                out, _ = vlm.generate(a.model, a.quant, SYS, c + [PROMPT.format(PREV=prev, A=s, B=s + a.win)], 160)
                line = out.strip().splitlines()[0] if out.strip() else ""
                lines.append(line); prev = line
            s += a.win
        f.write_text("\n".join(lines), encoding="utf-8")
        print(v, f"{time.time() - t0:.0f}s", len(lines), "lines", flush=True)


if __name__ == "__main__":
    main()
