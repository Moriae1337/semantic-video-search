from __future__ import annotations

from svs.config import Settings
from svs.schemas import VectorMode


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model_tag == "ViT-B-32/laion2b_s34b_b79k"
    assert s.vector_mode is VectorMode.segment
    assert s.collection_name == "svs__vit_b_32_laion2b_s34b_b79k__segment"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SVS_SEGMENT_SECONDS", "2.5")
    monkeypatch.setenv("SVS_MODEL_NAME", "ViT-L-14")
    monkeypatch.setenv("SVS_MODEL_PRETRAINED", "openai")
    monkeypatch.setenv("SVS_VECTOR_MODE", "frame")
    s = Settings(_env_file=None)
    assert s.segment_seconds == 2.5
    assert s.collection_name == "svs__vit_l_14_openai__frame"


def test_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SVS_FRAMES_PER_SEGMENT=8\nSVS_DATA_DIR=/mnt/big\n")
    s = Settings(_env_file=env)
    assert s.frames_per_segment == 8
    assert s.frames_dir.as_posix() == "/mnt/big/frames"
