"""
Index a YouTube video into the YouTube RAG backend before testing it.

The backend (a Flask server.py) answers questions
about whatever transcript was last sent to POST /save_transcript. This script sends one:

    # fetch the transcript from YouTube (pip install youtube-transcript-api)
    python -m examples.youtube_rag.index_video --video dQw4w9WgXcQ --server http://localhost:7860
    # or use a saved transcript: JSON list of {"text", "start"} (seconds) or {"text", "time"} ("M:SS")
    python -m examples.youtube_rag.index_video --transcript-file transcript.json

It also writes transcript_<id>.json, the source of truth for writing golden answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fetch_transcript(video_id: str, languages: list[str]) -> list[dict]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        sys.exit("Install it first: pip install youtube-transcript-api")
    api = YouTubeTranscriptApi
    if hasattr(api, "get_transcript"):  # < 1.0
        raw = api.get_transcript(video_id, languages=languages)
    else:  # >= 1.0
        raw = api().fetch(video_id, languages=languages).to_raw_data()
    return [{"text": r["text"], "start": r["start"]} for r in raw]


def normalize(rows: list[dict]) -> list[dict]:
    """Backend format: [{"text": ..., "time": "M:SS"}], like the Chrome extension sends."""
    out = []
    for r in rows:
        text = str(r.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue
        time = r.get("time") or fmt_time(float(r.get("start", 0)))
        out.append({"text": text, "time": str(time)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="YouTube video id (the part after v=)")
    src.add_argument("--transcript-file", help="Saved transcript JSON")
    p.add_argument("--server", default="http://localhost:7860", help="Backend base URL")
    p.add_argument("--languages", default="en", help="Comma-separated transcript languages to try")
    args = p.parse_args(argv)

    if args.video:
        rows = fetch_transcript(args.video, args.languages.split(","))
        name = args.video
    else:
        rows = json.loads(Path(args.transcript_file).read_text())
        name = Path(args.transcript_file).stem
    transcript = normalize(rows)
    if not transcript:
        sys.exit("Transcript is empty")

    print(f"Transcript: {len(transcript)} segments, last at {transcript[-1]['time']}")
    if args.video:  # keep a copy: the source of truth for golden answers
        saved = Path(f"transcript_{name}.json")
        saved.write_text(json.dumps(transcript, indent=1, ensure_ascii=False))
        print(f"Saved {saved}")

    resp = requests.post(args.server.rstrip("/") + "/save_transcript", json={"transcript": transcript}, timeout=600)
    print(f"/save_transcript -> HTTP {resp.status_code}: {resp.text[:200]}")
    return 0 if resp.ok else 1


if __name__ == "__main__":
    sys.exit(main())
