"""Real model tests: downloads ViT-B-32 once (~600 MB) and checks semantics on CPU."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open_clip")
pytestmark = pytest.mark.slow

from svs.embeddings import ClipEmbedder, segments_to_points  # noqa: E402
from svs.schemas import VectorMode  # noqa: E402

COLORS = ["red", "green", "blue", "yellow"]


@pytest.fixture(scope="module")
def emb() -> ClipEmbedder:
    return ClipEmbedder("ViT-B-32", "laion2b_s34b_b79k", device="cpu", batch_size=4)


def test_shapes_and_norms(emb, fixtures_dir):
    assert emb.dim == 512
    imgs = emb.embed_images([fixtures_dir / "images/red_0.png", fixtures_dir / "images/blue_0.png"])
    txt = emb.embed_text(["a dog", "a cat", "a car"])
    assert imgs.shape == (2, 512) and txt.shape == (3, 512)
    assert imgs.dtype == np.float32
    assert np.allclose(np.linalg.norm(imgs, axis=1), 1, atol=1e-4)
    assert np.allclose(np.linalg.norm(txt, axis=1), 1, atol=1e-4)


def test_empty_inputs(emb):
    assert emb.embed_images([]).shape == (0, 512)
    assert emb.embed_text([]).shape == (0, 512)


def test_batching_does_not_change_results(emb, fixtures_dir):
    paths = [fixtures_dir / f"images/{c}_{i}.png" for c in COLORS for i in range(2)]
    a = emb.embed_images(paths)  # batch_size=4 -> 2 batches
    emb.batch_size = 64
    try:
        b = emb.embed_images(paths)
    finally:
        emb.batch_size = 4
    assert np.allclose(a, b, atol=1e-4)


def test_text_finds_matching_color_image(emb, fixtures_dir):
    imgs = emb.embed_images([fixtures_dir / f"images/{c}_0.png" for c in COLORS])
    txt = emb.embed_text([f"a {c} image" for c in COLORS])
    sims = txt @ imgs.T
    assert list(sims.argmax(axis=1)) == [0, 1, 2, 3]


def test_search_over_fixture_segments(emb, segments, videos, fixtures_dir):
    """Mini end-to-end: embed fixture segments, query by text, top hit is the right video."""
    pts = segments_to_points(emb, segments, VectorMode.segment, root=fixtures_dir)
    matrix = np.stack([p.vector for p in pts])
    by_title = {v.video_id: v.title for v in videos}
    for color in COLORS:
        q = emb.embed_text([f"a {color} screen"])[0]
        best = pts[int(np.argmax(matrix @ q))]
        assert by_title[best.payload["video_id"]] == f"{color} demo"
