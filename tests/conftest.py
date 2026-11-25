from __future__ import annotations

import json
from pathlib import Path

import pytest

from svs.schemas import SearchResponse, Segment, Video

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def videos() -> list[Video]:
    data = json.loads((FIXTURES / "videos.json").read_text(encoding="utf-8"))
    return [Video.model_validate(v) for v in data]


@pytest.fixture
def segments() -> list[Segment]:
    data = json.loads((FIXTURES / "segments.json").read_text(encoding="utf-8"))
    return [Segment.model_validate(s) for s in data]


@pytest.fixture
def search_response() -> SearchResponse:
    data = json.loads((FIXTURES / "search_response.json").read_text(encoding="utf-8"))
    return SearchResponse.model_validate(data)
