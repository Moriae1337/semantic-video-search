# Interfaces between zones (L1)

Single source of truth in code: [`src/svs/schemas.py`](../src/svs/schemas.py) and
[`src/svs/config.py`](../src/svs/config.py). This file explains them in words.
Changes go through a PR that tags the owners of the affected zones.

## Data flow

```
 video files ──A──> Video + list[Segment] (+ frames, thumbnail, clip on disk)
                         │
                         ├──B──> embeddings.segments_to_points(...) -> list[IndexPoint]
                         │                                  (id, vector, payload)
                         │
                         └──C──> Qdrant upsert(IndexPoint)      text query
                                       │                           │
                                       │<──── B: embed_text ───────┘
                                       v
                                 SearchResponse(list[SearchResult]) ──D──> UI
```

| From → To | What | Python type |
|---|---|---|
| A → B | Segments with paths to frames on disk | `list[Segment]` (+ `Video`) |
| B → C | Vectors to write | `list[IndexPoint]` from `svs.embeddings.segments_to_points` |
| C ↔ B | Query vector at search time | `Embedder.embed_text([query])[0]` |
| C → D | Search results over HTTP | `SearchResponse` (JSON) |

## IDs

- **`video_id`** = `make_video_id(path)`: first 16 hex chars of sha1 over the **file name**.
  The same dataset gets the same ids on any machine. File names must be unique in the dataset.
- **`segment_id`** = `make_segment_id(video_id, index)`: UUIDv5 string. Used directly as the
  Qdrant point id (Qdrant only accepts unsigned ints or UUIDs). Deterministic, so re-indexing
  overwrites points instead of duplicating them.
- **Frame point id** (frame mode only) = `make_frame_point_id(segment_id, frame_index)`.

## Segment

| Field | Meaning |
|---|---|
| `segment_id`, `video_id`, `index` | see above; `index` counts from 0 inside the video |
| `start`, `end` | seconds from the video start, `0 <= start < end` |
| `frame_paths`, `frame_times` | frames given to the model and their timestamps; same length, each time inside `[start, end]` |
| `thumbnail_path` | one image for the results list |
| `clip_path` | short mp4 of the segment, or `null` if clips are cut on request |

Paths are relative to `SVS_DATA_DIR`. All models are `extra="forbid"`: unknown fields are errors.

## Vectors

- Float32, L2-normalized: **cosine similarity = dot product**. Qdrant distance: `Cosine`.
- Dimension: `Embedder.dim` (512 for ViT-B-32, 768 for ViT-L-14, etc.). Read it, don't hardcode.
- Two vector modes (`SVS_VECTOR_MODE`), so B4 can compare them:
  - `segment` (default): one vector per segment = normalized mean of its frame vectors.
  - `frame`: one vector per frame; C groups hits by `segment_id` and keeps the best score.

## Qdrant

- **One collection per model and vector mode**: `Settings.collection_name`, e.g.
  `svs__vit_b_32_laion2b_s34b_b79k__segment`. Different models have different dimensions, so
  they can't share a collection. The `model` payload field is also stored as a safety check.
- Payload (`Segment.to_payload(model)`):

```json
{
  "segment_id": "uuid", "video_id": "16 hex", "index": 0,
  "start": 0.0, "end": 5.0,
  "thumbnail_path": "frames/.../thumb.jpg", "clip_path": null,
  "model": "ViT-B-32/laion2b_s34b_b79k"
}
```

  Frame mode adds `frame_index` and `frame_time`. Index `video_id` as a keyword field for filters.

## HTTP API (draft for C3 and D3)

| Method | Path | Body / params | Response |
|---|---|---|---|
| `POST` | `/search` | `SearchRequest` `{query, k=10, video_id?}` | `SearchResponse` |
| `POST` | `/videos` | multipart file | `{video_id, status}` |
| `GET` | `/videos/{video_id}` | | `{video_id, status: queued\|processing\|done\|failed, error?}` |
| `GET` | `/media/thumbnails/{segment_id}.jpg` | | image |
| `GET` | `/media/clips/{segment_id}.mp4` | | video |

Errors: `422` invalid request (e.g. empty query), `404` unknown video, `503` Qdrant unavailable.
Example response: [`tests/fixtures/search_response.json`](../tests/fixtures/search_response.json).

## Fixtures

`tests/fixtures/` lets every zone work without the others (regenerate with
`python scripts/make_fixtures.py`):

- `images/*.png`: 4 "videos" of solid colours, 2 frames each
- `videos.json`, `segments.json`: 4 videos × 2 segments, valid per the contract
- `search_response.json`: what `/search` returns, for building the UI

## Open questions

These were listed in the backlog for discussion in this PR; the proposed answers are above and in
[`decisions.md`](decisions.md). Owners, please comment if you disagree.
