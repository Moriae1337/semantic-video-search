"""Regenerate tests/fixtures/: images, videos.json, segments.json, search_response.json.

The fixtures let every zone work without the others:
- A: example of what segmentation must output
- B/C: frames to embed and segments to index
- D: a search response to build the UI against

Run: python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from svs.schemas import (
    SearchResponse,
    SearchResult,
    Segment,
    Video,
    make_segment_id,
    make_video_id,
)

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
IMG = FIX / "images"

# Each "video" is a solid colour; frames differ slightly so they are not identical.
COLORS = {
    "red": (220, 30, 30),
    "green": (30, 170, 50),
    "blue": (30, 60, 220),
    "yellow": (240, 220, 30),
}


def make_images() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    for name, rgb in COLORS.items():
        for f in range(2):
            im = Image.new("RGB", (224, 224), rgb)
            d = ImageDraw.Draw(im)
            shade = tuple(max(0, c - 25) for c in rgb)
            d.rectangle([20 + 40 * f, 20, 120 + 40 * f, 120], fill=shade)
            im.save(IMG / f"{name}_{f}.png")


def make_json() -> None:
    videos, segments = [], []
    for name in COLORS:
        file = f"{name}_demo.mp4"
        vid = make_video_id(file)
        videos.append(
            Video(
                video_id=vid,
                path=f"sample/{file}",
                duration=10.0,
                fps=25.0,
                width=224,
                height=224,
                title=f"{name} demo",
                caption=f"a plain {name} screen",
            )
        )
        for idx in range(2):
            start, end = idx * 5.0, (idx + 1) * 5.0
            segments.append(
                Segment(
                    segment_id=make_segment_id(vid, idx),
                    video_id=vid,
                    index=idx,
                    start=start,
                    end=end,
                    frame_paths=[f"images/{name}_0.png", f"images/{name}_1.png"],
                    frame_times=[start + 1.25, start + 3.75],
                    thumbnail_path=f"images/{name}_0.png",
                    clip_path=None,
                )
            )

    blue = [s for s in segments if s.video_id == make_video_id("blue_demo.mp4")]
    response = SearchResponse(
        query="a blue screen",
        model="ViT-B-32/laion2b_s34b_b79k",
        took_ms=12.3,
        results=[
            SearchResult(
                segment_id=s.segment_id,
                video_id=s.video_id,
                video_title="blue demo",
                start=s.start,
                end=s.end,
                score=round(0.31 - 0.01 * i, 4),
                thumbnail_url=f"/media/thumbnails/{s.segment_id}.jpg",
                clip_url=f"/media/clips/{s.segment_id}.mp4",
            )
            for i, s in enumerate(blue)
        ],
    )

    def dump(name: str, obj) -> None:
        (FIX / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    dump("videos.json", [v.model_dump(mode="json") for v in videos])
    dump("segments.json", [s.model_dump(mode="json") for s in segments])
    dump("search_response.json", response.model_dump(mode="json"))


if __name__ == "__main__":
    make_images()
    make_json()
    print(f"fixtures written to {FIX}")
