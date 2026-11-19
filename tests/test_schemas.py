from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from svs.schemas import (
    SearchRequest,
    Segment,
    make_frame_point_id,
    make_segment_id,
    make_video_id,
)


def _seg(**kw) -> Segment:
    base = dict(
        segment_id=make_segment_id("v", 0),
        video_id="v",
        index=0,
        start=0.0,
        end=5.0,
        frame_paths=["a.png", "b.png"],
        frame_times=[1.0, 4.0],
        thumbnail_path="a.png",
    )
    base.update(kw)
    return Segment(**base)


def test_video_id_depends_only_on_file_name():
    assert make_video_id("/data/x/clip.mp4") == make_video_id("other/dir/clip.mp4")
    assert make_video_id("a.mp4") != make_video_id("b.mp4")
    assert len(make_video_id("a.mp4")) == 16


def test_segment_id_is_deterministic_uuid():
    a = make_segment_id("vid", 3)
    assert a == make_segment_id("vid", 3)
    assert a != make_segment_id("vid", 4)
    assert a != make_segment_id("vid2", 3)
    uuid.UUID(a)  # valid Qdrant point id


def test_frame_point_ids_unique_per_frame():
    sid = make_segment_id("vid", 0)
    ids = {make_frame_point_id(sid, i) for i in range(10)}
    assert len(ids) == 10
    assert sid not in ids


def test_valid_segment():
    s = _seg()
    assert s.end - s.start == 5.0


@pytest.mark.parametrize(
    "kw",
    [
        dict(end=0.0),  # end == start
        dict(start=6.0, frame_times=[6.5, 7.0]),  # end < start
        dict(start=-1.0),
        dict(frame_times=[1.0]),  # length mismatch
        dict(frame_times=[1.0, 9.0]),  # frame outside segment
        dict(frame_paths=[], frame_times=[]),
        dict(index=-1),
    ],
)
def test_invalid_segment(kw):
    with pytest.raises(ValidationError):
        _seg(**kw)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        _seg(unknown=1)


def test_payload_has_contract_fields():
    p = _seg().to_payload("ViT-B-32/laion2b_s34b_b79k")
    assert set(p) == {
        "segment_id",
        "video_id",
        "index",
        "start",
        "end",
        "thumbnail_path",
        "clip_path",
        "model",
    }
    assert p["model"] == "ViT-B-32/laion2b_s34b_b79k"


def test_search_request_limits():
    assert SearchRequest(query="dog").k == 10
    for bad in (dict(query=""), dict(query="x", k=0), dict(query="x", k=101)):
        with pytest.raises(ValidationError):
            SearchRequest(**bad)


def test_fixtures_are_valid_and_consistent(videos, segments, search_response, fixtures_dir):
    video_ids = {v.video_id for v in videos}
    assert len(videos) == 4 and len(segments) == 8
    assert all(s.video_id in video_ids for s in segments)
    assert len({s.segment_id for s in segments}) == len(segments)
    for s in segments:
        assert s.segment_id == make_segment_id(s.video_id, s.index)
        for p in s.frame_paths:
            assert (fixtures_dir / p).is_file()
    seg_ids = {s.segment_id for s in segments}
    assert all(r.segment_id in seg_ids for r in search_response.results)
