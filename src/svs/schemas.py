"""Data contract shared by all zones (L1).

Flow: A produces `Video` + `Segment`s -> B embeds them -> C stores them in Qdrant
and returns `SearchResult`s -> D shows them.

Changing anything here needs a PR that tags the owners of the affected zones.
"""

from __future__ import annotations

import hashlib
import uuid
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Fixed namespace so segment ids are the same on every machine and every run.
SVS_NAMESPACE = uuid.UUID("6f1c2a52-6a52-4c5e-9d43-3b6f0c1d8e01")


def make_video_id(path: str | Path) -> str:
    """Stable video id: first 16 hex chars of sha1 over the file name (not the full path).

    Using only the name means the same dataset gives the same ids on any machine,
    regardless of where it is stored on disk.
    """
    name = Path(path).name
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def make_segment_id(video_id: str, index: int) -> str:
    """Stable segment id as a UUID string, usable directly as a Qdrant point id.

    Deterministic, so re-indexing the same video overwrites points instead of duplicating them.
    """
    return str(uuid.uuid5(SVS_NAMESPACE, f"{video_id}:{index}"))


def make_frame_point_id(segment_id: str, frame_index: int) -> str:
    """Stable Qdrant point id for one frame vector (used in `VectorMode.frame`)."""
    return str(uuid.uuid5(SVS_NAMESPACE, f"{segment_id}:f{frame_index}"))


class VectorMode(str, Enum):
    """How a segment is represented in the index."""

    segment = "segment"  # one vector per segment (mean of its frame vectors)
    frame = "frame"  # one vector per frame, search results grouped by segment


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Video(_Model):
    """A source video file. Produced by zone A."""

    video_id: str = Field(description="make_video_id(path)")
    path: str = Field(description="Path to the source file, relative to SVS_DATA_DIR")
    duration: float = Field(ge=0, description="Seconds")
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    title: str | None = None
    caption: str | None = Field(default=None, description="Dataset caption, ground truth for B4")


class Segment(_Model):
    """A time range of a video: the unit of search. Produced by zone A, embedded by B."""

    segment_id: str = Field(description="make_segment_id(video_id, index), a UUID string")
    video_id: str
    index: int = Field(ge=0, description="Position of the segment inside its video, from 0")
    start: float = Field(ge=0, description="Seconds from the start of the video")
    end: float = Field(description="Seconds, > start")
    frame_paths: list[str] = Field(min_length=1, description="Frames given to the model")
    frame_times: list[float] = Field(min_length=1, description="Timestamp of each frame, seconds")
    thumbnail_path: str
    clip_path: str | None = Field(default=None, description="None if clips are cut on request")

    @model_validator(mode="after")
    def _check(self) -> Segment:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        if len(self.frame_paths) != len(self.frame_times):
            raise ValueError("frame_paths and frame_times must have the same length")
        for t in self.frame_times:
            if not (self.start <= t <= self.end):
                raise ValueError(f"frame time {t} is outside [{self.start}, {self.end}]")
        return self

    def to_payload(self, model: str) -> dict:
        """Qdrant payload for this segment. `model` is the embedding model tag."""
        return {
            "segment_id": self.segment_id,
            "video_id": self.video_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "thumbnail_path": self.thumbnail_path,
            "clip_path": self.clip_path,
            "model": model,
        }


class SearchRequest(_Model):
    """Body of POST /search."""

    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=100)
    video_id: str | None = Field(default=None, description="Optional filter: search one video")


class SearchResult(_Model):
    """One hit returned by the search API. Produced by zone C, shown by D."""

    segment_id: str
    video_id: str
    video_title: str | None = None
    start: float = Field(ge=0)
    end: float
    score: float = Field(description="Cosine similarity, higher is better")
    thumbnail_url: str
    clip_url: str
    matched_frame_time: float | None = Field(
        default=None, description="Best-matching frame (VectorMode.frame only)"
    )


class SearchResponse(_Model):
    """Response of POST /search."""

    query: str
    model: str
    took_ms: float = Field(ge=0)
    results: list[SearchResult]
