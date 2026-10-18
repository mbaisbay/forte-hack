"""Probe: mixed denominations in one stack (231.46.1). Qwen on grounded desk crops -> data/denom/<v>.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import frames, state, ml, detect
from evaluate import video_path

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    for v in ml.VIDS:
        o = ROOT / f"data/denom/{v}.json"
        if o.exists():
            continue
        fr = frames.sample(video_path(v), fps=1, max_frames=60, width=464, overlay=False)
        g = json.loads((ROOT / f"data/ground/{v}.json").read_text(encoding="utf-8"))
        A = state.ask_frames({"model": detect.QWEN, "quant": "none", "batch": 4, "state_tokens": 20,
                              "state_system": "denom_system.txt", "state_prompt": "denom.txt"}, detect.desk_crops(fr, g["desk"]))
        o.write_text(json.dumps([a.get("mixed", 0) for a in A]), encoding="utf-8")
        print(v, "V" if ml.GT[v] else "-", sum(a.get("mixed", 0) for a in A), len(A), flush=True)
