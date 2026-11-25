"""Text and image embeddings in one vector space (B2).

The rest of the system talks to the `Embedder` protocol only, so the model can be
switched through config (SVS_MODEL_NAME / SVS_MODEL_PRETRAINED) without code changes.
All returned vectors are float32 and L2-normalized, so dot product == cosine similarity.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

from svs.schemas import Segment, VectorMode, make_frame_point_id

if TYPE_CHECKING:
    from svs.config import Settings

ImageInput = str | Path | Image.Image


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def model_tag(self) -> str: ...

    def embed_images(self, images: Sequence[ImageInput]) -> np.ndarray: ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def aggregate_frames(frame_vectors: np.ndarray) -> np.ndarray:
    """Segment vector = normalized mean of its (normalized) frame vectors."""
    frame_vectors = np.asarray(frame_vectors, dtype=np.float32)
    if frame_vectors.ndim != 2 or len(frame_vectors) == 0:
        raise ValueError("expected a non-empty (n_frames, dim) array")
    return l2_normalize(frame_vectors.mean(axis=0))


def _batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _load_image(img: ImageInput) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    with Image.open(img) as im:
        return im.convert("RGB")


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but CUDA is not available")
    return device


class ClipEmbedder:
    """CLIP / SigLIP models through open_clip. Any `open_clip.list_pretrained()` pair works."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            self._dim = int(self.model.encode_text(self.tokenizer(["x"]).to(self.device)).shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_tag(self) -> str:
        return f"{self.model_name}/{self.pretrained}"

    def _autocast(self):
        # fp16 on GPU for speed; full precision on CPU.
        return self._torch.autocast(
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
            enabled=self.device.startswith("cuda"),
        )

    def embed_images(self, images: Sequence[ImageInput]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out = []
        for batch in _batches(images, self.batch_size):
            pixels = torch.stack([self.preprocess(_load_image(im)) for im in batch]).to(self.device)
            with torch.no_grad(), self._autocast():
                out.append(self.model.encode_image(pixels).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out = []
        for batch in _batches(list(texts), self.batch_size):
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad(), self._autocast():
                out.append(self.model.encode_text(tokens).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))


def get_embedder(settings: Settings | None = None) -> ClipEmbedder:
    """Build the embedder described by config. Load it once and reuse it."""
    if settings is None:
        from svs.config import get_settings

        settings = get_settings()
    return ClipEmbedder(
        model_name=settings.model_name,
        pretrained=settings.model_pretrained,
        device=settings.device,
        batch_size=settings.embed_batch_size,
    )


@dataclass(frozen=True)
class IndexPoint:
    """What zone C writes to Qdrant: point id, vector and payload."""

    id: str
    vector: np.ndarray
    payload: dict


def segments_to_points(
    embedder: Embedder,
    segments: Sequence[Segment],
    mode: VectorMode = VectorMode.segment,
    root: str | Path | None = None,
) -> list[IndexPoint]:
    """Embed the frames of all segments and turn them into index points.

    `segment` mode: one point per segment, id = segment_id, vector = mean of frame vectors.
    `frame` mode: one point per frame, payload also has `frame_time`.
    `root` is prepended to relative frame paths.
    """
    root = Path(root) if root is not None else None
    paths = [
        (root / p if root is not None else Path(p)) for seg in segments for p in seg.frame_paths
    ]
    vectors = embedder.embed_images(paths)

    points: list[IndexPoint] = []
    offset = 0
    for seg in segments:
        n = len(seg.frame_paths)
        frame_vecs = vectors[offset : offset + n]
        offset += n
        payload = seg.to_payload(embedder.model_tag)
        if mode == VectorMode.segment:
            points.append(IndexPoint(seg.segment_id, aggregate_frames(frame_vecs), payload))
        else:
            for i, (vec, t) in enumerate(zip(frame_vecs, seg.frame_times, strict=True)):
                points.append(
                    IndexPoint(
                        make_frame_point_id(seg.segment_id, i),
                        vec,
                        {**payload, "frame_index": i, "frame_time": t},
                    )
                )
    return points
