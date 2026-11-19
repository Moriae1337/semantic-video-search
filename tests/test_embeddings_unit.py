"""Embedding helpers tested with a fake model: fast, no model download needed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from svs.embeddings import aggregate_frames, l2_normalize, segments_to_points
from svs.schemas import VectorMode


class FakeEmbedder:
    dim = 8
    model_tag = "fake/0"

    def _vec(self, key: str) -> np.ndarray:
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self.dim)

    def embed_images(self, images):
        return l2_normalize(np.stack([self._vec(Path(p).name) for p in images]))

    def embed_text(self, texts):
        return l2_normalize(np.stack([self._vec(t) for t in texts]))


def test_l2_normalize():
    x = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(x[0], [0.6, 0.8])
    assert np.all(np.isfinite(x[1]))


def test_aggregate_frames_is_unit_mean_direction():
    v = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    agg = aggregate_frames(v)
    assert np.isclose(np.linalg.norm(agg), 1.0)
    assert np.allclose(agg, [2**-0.5, 2**-0.5])


def test_aggregate_frames_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_frames(np.zeros((0, 4)))


def test_segment_mode_one_point_per_segment(segments, fixtures_dir):
    pts = segments_to_points(FakeEmbedder(), segments, VectorMode.segment, root=fixtures_dir)
    assert len(pts) == len(segments)
    assert [p.id for p in pts] == [s.segment_id for s in segments]
    for p in pts:
        assert p.vector.shape == (8,)
        assert np.isclose(np.linalg.norm(p.vector), 1.0)
        assert p.payload["model"] == "fake/0"


def test_frame_mode_one_point_per_frame(segments, fixtures_dir):
    pts = segments_to_points(FakeEmbedder(), segments, VectorMode.frame, root=fixtures_dir)
    assert len(pts) == sum(len(s.frame_paths) for s in segments)
    assert len({p.id for p in pts}) == len(pts)
    first = pts[0]
    assert first.payload["frame_index"] == 0
    assert first.payload["frame_time"] == segments[0].frame_times[0]


def test_points_are_deterministic(segments, fixtures_dir):
    a = segments_to_points(FakeEmbedder(), segments, VectorMode.frame, root=fixtures_dir)
    b = segments_to_points(FakeEmbedder(), segments, VectorMode.frame, root=fixtures_dir)
    assert [p.id for p in a] == [p.id for p in b]
    assert all(np.array_equal(x.vector, y.vector) for x, y in zip(a, b, strict=True))
