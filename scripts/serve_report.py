"""Serve eval reports and the dev videos on localhost, so browsers can play and seek the clips.

Opening the report as a file:// page blocks the videos in Safari; over HTTP with range
requests both Safari and Chrome play them. Only reports/ and the videos folder are exposed.

Usage:
  python scripts/serve_report.py [--port 8765]
  then open http://127.0.0.1:8765/reports/eval_report_r3.html
"""

import argparse
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]


def build_app(reports: Path, videos: Path) -> Starlette:
    return Starlette(routes=[
        Mount("/reports", StaticFiles(directory=reports)),
        Mount("/open/videos", StaticFiles(directory=videos)),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--videos", type=Path, default=ROOT / "open" / "videos")
    args = ap.parse_args()
    uvicorn.run(build_app(args.reports, args.videos), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
