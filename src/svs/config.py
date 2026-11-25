"""All tunable parameters in one place. Set any of them via env or `.env` as SVS_<NAME>."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from svs.schemas import VectorMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SVS_", env_file=".env", extra="ignore")

    # data
    data_dir: Path = Path("data")
    sample_dir: Path = Path("data/sample")

    # segmentation (zone A)
    segment_seconds: float = Field(default=5.0, gt=0)
    frames_per_segment: int = Field(default=4, ge=1)
    num_workers: int = Field(default=4, ge=1)

    # embeddings (zone B)
    model_name: str = "ViT-B-32"
    model_pretrained: str = "laion2b_s34b_b79k"
    device: str = Field(default="auto", description="auto | cpu | cuda | cuda:0 ...")
    embed_batch_size: int = Field(default=64, ge=1)
    vector_mode: VectorMode = VectorMode.segment

    # storage and API (zone C)
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_path: Path | None = Field(
        default=None, description="If set, use embedded local Qdrant instead of qdrant_url"
    )
    api_base_url: str = "http://localhost:8000"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def model_tag(self) -> str:
        """Model id stored in the Qdrant payload, e.g. 'ViT-B-32/laion2b_s34b_b79k'."""
        return f"{self.model_name}/{self.model_pretrained}"

    @property
    def collection_name(self) -> str:
        """One collection per model and vector mode (different models have different dims)."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.model_tag.lower()).strip("_")
        return f"svs__{slug}__{self.vector_mode.value}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
