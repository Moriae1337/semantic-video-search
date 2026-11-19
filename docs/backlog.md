# Backlog: Semantic Video Search

4 epics (one per zone owner) + 20 issues.
Issues describe **why** and **what the outcome should be**. **How** to get there is up to the zone owner.

| Zone | Owner | Issues |
|---|---|---|
| Contract | group lead (B) | L1 |
| Data & Ingestion | A | A1–A4 |
| ML & Retrieval quality | B | B1–B5 |
| Search & Infrastructure | C | C1–C5 |
| Product & Story | D | D1–D5 |

## Working agreements

- **The zone owner** makes decisions within their zone on their own and briefly records them in `docs/decisions.md`: what was decided and why.
- **Shared interfaces** (`schemas.py`, API format, Qdrant payload) change only via a PR that tags the owners of the affected zones.
- **Review in a ring:** A → B → C → D → A. Respond to PRs within 24 hours.
- **Every 2–3 days**, post a short update in the pinned "Status" issue: done / doing / blocked by what.
- **Each owner** prepares the slides for their zone and presents it at the defense.
- **The owner may** split, merge, add or close issues in their epic. Large issues are best split into sub-issues.

## Milestones

| Milestone | Deadline | Goal |
|---|---|---|
| **M1: Skeleton** | 23.11 | The repo clones and runs, interfaces are agreed, data is on disk |
| **M2: End-to-end pipeline** | 28.11 | Video → segments → embeddings → Qdrant → search → UI works on sample data |
| **M3: Big Data, evaluation, defense** | 02.12 | Full dataset indexed, numbers collected, presentation ready |

## Labels

`zone:ingest` · `zone:ml` · `zone:search` · `zone:product` · `contract` · `epic`

---

## L1. Interface contract between zones
`contract` · **M1** · owner: group lead (B)

**Why.** We work asynchronously, so every zone must be able to start without waiting for the others. That requires a single agreement on what data passes between zones and in what format. Once this PR is merged, everyone works on their piece against stubs.

**Outcome**
- [ ] `src/svs/schemas.py`: `Video`, `Segment`, `SearchResult` models (pydantic)
- [ ] `src/svs/config.py`: all parameters (segment length, frames per segment, model, Qdrant URL, data paths) are set via env/`.env`
- [ ] `docs/interfaces.md`: data flow between zones, Qdrant payload, API contract (requests and responses)
- [ ] Fixtures in `tests/fixtures/`: sample segments, a search response, a few images, so each zone can work without the others
- [ ] PR reviewed by all three other owners (or silent consent by the deadline)

**Open questions to discuss in the PR**
- ID format for videos and segments. A Qdrant point id can only be an integer or a UUID.
- One vector per segment or one per frame. We can support both so B4 can compare them.
- How to mark the model in the data: a separate collection per model or a payload field.

---

## Epic A: Data & Ingestion
`epic` `zone:ingest` · owner: **A**

**Goal.** Turn an arbitrary set of video files into segments with frames, thumbnails and clips, quickly and reliably, for tens of hours of video. This zone provides the core big data argument: processing a large volume of unstructured data in a way that scales.

**Epic success criteria**
- 50+ hours of video indexed
- A speedup chart against the number of processes
- Broken or unusual files don't crash the pipeline
- Slides for stage "3. Dataset preparation"

### A1. Dataset: selection, download, sample subset
`zone:ingest` · **M1**

**Why.** Everyone needs real data for development. Quality evaluation (B4) needs text descriptions of the videos as ground truth. The demo needs long videos to show finding a moment inside a clip.

**Outcome**
- [ ] Main dataset chosen, with a short justification
- [ ] Download script reproduces from scratch on a clean machine
- [ ] `data/sample/` (~20 videos) for fast development on CPU
- [ ] A few long videos (5+ minutes) for the demo
- [ ] Dataset statistics: number of videos, total duration, resolution, size on disk

**Owner's call:** source (candidates: MSR-VTT with clip captions, ActivityNet Captions with timestamps, Pexels with a free license), size, where to store the full dataset and how to hand it to B for the final indexing.

### A2. Video segmentation: segments, frames, thumbnails, clips
`zone:ingest` · **M2** · depends on: L1, A1

**Why.** A segment is the unit of search: it's what the system returns to the user, together with timestamps. Each segment needs frames for the model (B), a thumbnail for the results list and a short clip for playback (D).

**Outcome**
- [ ] A `video → list[Segment]` function per the contract, with correct `start`/`end` for videos with different fps and duration
- [ ] For each segment: N frames, 1 thumbnail, a clip (or a way to produce it on the fly), with paths stored in `Segment`
- [ ] Edge-case tests: video shorter than a segment, zero duration, variable fps, no audio or an odd codec
- [ ] An estimate of how much disk space 1 hour of video takes after processing

**Owner's call**
- Segmentation method: fixed-length chunks, scene detection (PySceneDetect), or both switchable via config. Fixed chunks are the easier start.
- Frame sampling strategy: uniform, first/middle/last, on scene change.
- Tools: FFmpeg, OpenCV, PyAV.
- Frame and thumbnail format and size; clips generated upfront or on request.

### A3. Parallel and fault-tolerant processing of many videos
`zone:ingest` · **M2** · depends on: A2 · works with B on B3

**Why.** This is the core big data aspect. Processing tens of hours of video must scale across all CPU cores, survive failures in individual files and not redo finished work.

**Outcome**
- [ ] Processes a folder of videos in N processes or workers, with N set in config
- [ ] An error in one file is logged and doesn't stop the rest; a report of failed files at the end
- [ ] Re-running doesn't reprocess videos that are already done
- [ ] Progress is visible (videos done, throughput, estimated time remaining)

**Owner's call:** multiprocessing, concurrent.futures, Ray, Dask; NVDEC hardware decoding. How frames are handed to the GPU stage is decided together with B in B3.

### A4. Ingestion scaling benchmark
`zone:ingest` · **M3** · depends on: A3

**Why.** We need numbers for the presentation that show parallel processing has a real effect, and honestly explain where the bottleneck is.

**Outcome**
- [ ] Processing time of a fixed video set with 1, 2, 4… N processes
- [ ] A speedup chart and an explanation of the bottleneck (decoding, disk, writing)
- [ ] Benchmark script in `scripts/` so the numbers are reproducible

---

## Epic B: ML & Retrieval quality
`epic` `zone:ml` · owner: **B**

**Goal.** Map frames and text queries into one vector space so that search finds the right fragments, and prove it with numbers. This zone also owns the GPU part of the pipeline and the final indexing, since only B has a GPU.

**Epic success criteria**
- Recall@K on a labeled query set
- At least 3 models or configurations compared
- Known throughput (frames/s) on GPU and on CPU
- Full dataset indexed
- Slides for stage "4. Methods and models"

### B1. GPU and CPU environment
`zone:ml` · **M1**

**Why.** RTX 50-series GPUs need PyTorch with CUDA 12.8+, while the rest of the team works on CPU. Both setups must work on the first try, otherwise half the team gets stuck on installation.

**Outcome**
- [ ] Install instructions in the README for CUDA and for CPU
- [ ] `scripts/check_env.py`: versions, device, embedding of one image and one text
- [ ] Verified on at least one GPU machine and one CPU machine

### B2. Frame, text and segment embeddings
`zone:ml` · **M2** · depends on: L1, B1

**Why.** This is the core of semantic search: query text and video frames must land in the same vector space. The rest of the system should go through one interface and not depend on a specific model. A segment contains several frames, and how they are aggregated directly affects search quality.

**Outcome**
- [ ] `embed_images(...)` and `embed_text(...)` with batching, normalized vectors and an exposed dimension (for creating the Qdrant collection)
- [ ] The model is switched via config, without code changes
- [ ] Works on CPU (slowly) and on GPU
- [ ] At least one way to build a segment representation from its frames
- [ ] The segment representation decision is documented and agreed with C (how it maps to Qdrant and how to search it)

**Owner's call**
- Library: open_clip (CLIP and SigLIP behind one API), transformers or another.
- fp16, batch size, model caching.
- Aggregation: mean-pooling frames into one vector, one vector per frame with results grouped by segment, other approaches.

### B3. End-to-end indexing pipeline: CPU decoding → GPU → Qdrant
`zone:ml` · **M2** · depends on: A3, B2, C2 · works with A and C

**Why.** This is where all zones meet. The GPU should stay busy instead of waiting on video decoding. This pipeline is what the architecture slide shows, and it's where the effect of parallel processing is most visible.

**Outcome**
- [ ] A single command (CLI script) indexes a folder of videos: frames from A's parallel workers → batched embeddings on GPU → writes via C's storage layer
- [ ] Also works on CPU (on `data/sample/`) so others can check their changes
- [ ] Throughput measured for each stage and/or GPU utilization
- [ ] Pipeline diagram in `docs/` (for the architecture slide)

**Owner's call (together with A):** queues between processes, exchange via disk or memory, number of workers per stage.

### B4. Search quality evaluation and model comparison
`zone:ml` · **M2 → M3** · depends on: B3, C3 · D helps with queries

**Why.** Without a metric we can neither choose a model nor justify our decisions at the defense. This is the main analytical result for stage 4 of the presentation.

**Outcome**
- [ ] *(M2)* A set of test queries with correct answers: from dataset captions and/or manual labeling (20–30 queries for the demo videos)
- [ ] *(M2)* A script that computes Recall@K (and/or MRR) through the real search, and a baseline for the first model
- [ ] *(M3)* A table: model/settings × quality × indexing speed × query speed × memory
- [ ] *(M3)* A few examples of good and bad queries, with explanations
- [ ] *(M3)* A conclusion on what was chosen for the final system and why

**Owner's call:** which models (e.g. CLIP ViT-B/32, ViT-L/14, SigLIP) and which parameters to compare (segment length, number of frames, aggregation).

### B5. Final indexing of the full dataset
`zone:ml` · **M3** · depends on: B3, B4

**Why.** The demo and benchmarks at the defense must run on the full data volume. We'll show the size figures on the slides.

**Outcome**
- [ ] Full dataset indexed with the chosen configuration
- [ ] Recorded: volume (hours, segments, vectors), indexing time, index size on disk
- [ ] A way to hand the ready index to the team (Qdrant snapshot or similar) so C and D can work with the full index without a GPU

---

## Epic C: Search & Infrastructure
`epic` `zone:search` · owner: **C**

**Goal.** Build a system that comes up with one command, stores the vectors and answers search queries quickly through an API. This zone owns the infrastructure side of big data: the vector index and fast similarity search over a large number of fragments.

**Epic success criteria**
- The system can be brought up from a clean clone by following the instructions
- Known p95 search latency on the full index
- API documented (OpenAPI)
- Slides for stage "5. Solution" (architecture)

### C1. Repo skeleton, CI and Qdrant
`zone:search` · **M1**

**Why.** A shared foundation for everyone: the same structure, the same dependencies, automatic checks on every PR and a vector DB anyone can run locally. In an async team a broken `main` might only be noticed a day later, so CI matters a lot here.

**Outcome**
- [ ] Package layout `src/svs/...`, dependency manager, `.gitignore` (data, indexes, models)
- [ ] GitHub Actions: linter + CPU tests on every PR
- [ ] `CONTRIBUTING.md`: branches, PRs, review, the agreements from this file
- [ ] Qdrant comes up with one command (Docker Compose), with persistent storage
- [ ] A no-Docker option (e.g. qdrant-client local mode), switchable via config

**Owner's call:** uv or pip, ruff, pre-commit, Python version.

### C2. Storage layer: writing and searching vectors
`zone:search` · **M2** · depends on: L1, B2

**Why.** Isolates the rest of the code from Qdrant and provides one interface for indexing and search. If we ever want to swap Qdrant (e.g. for FAISS as a comparison), only this layer changes.

**Outcome**
- [ ] Creating a collection for a specific model
- [ ] Batched upsert of segments with payload, idempotent (re-writing doesn't duplicate data)
- [ ] Top-K vector search with optional filters (e.g. by video)
- [ ] Tests on random vectors, without a model

**Owner's call:** HNSW parameters, quantization, collection layout for multiple models, write batch size.

### C3. FastAPI: indexing, search and media
`zone:search` · **M2** · depends on: B2, C2; B3 for indexing via the API

**Why.** The API is the main entry point to the system: for the UI, the quality evaluation script and the demo. Through it the user uploads a video, sees when it becomes searchable, and gets fragments with timestamps for a text query, ready to watch.

**Outcome**
- [ ] Search endpoint: query + K → results per the contract (segment, video, `start`/`end`, score, thumbnail and clip URLs)
- [ ] Thumbnails and clips are served at the URLs in the response
- [ ] The model is loaded once at startup, not on every request
- [ ] Video upload endpoint with background processing, and a processing status endpoint
- [ ] Error handling: empty query, video not found, Qdrant unavailable
- [ ] OpenAPI documentation with examples

**Owner's call:** BackgroundTasks or a separate worker for processing, router structure, CORS, file size limits.

### C4. Search performance benchmark
`zone:search` · **M3** · depends on: B5

**Why.** Show that search stays fast when there are very many fragments. This is the second half of the big data argument, after ingestion scaling (A4).

**Outcome**
- [ ] Latency (p50/p95) against index size, with query embedding and Qdrant search measured separately
- [ ] Exact vs HNSW search and/or different index parameters compared (speed vs accuracy)
- [ ] Charts for the presentation, a script to reproduce them

**Owner's call:** how to get large indexes (real data + synthetic vectors for scale), which parameters to vary.

### C5. One-command launch of the whole system
`zone:search` · **M3** · depends on: C3, D3

**Why.** The demo at the defense and checking by the instructor shouldn't depend on the setup of one particular laptop.

**Outcome**
- [ ] From a clean clone, following the instructions, Qdrant + API + UI come up
- [ ] A ready index (from B5) can be loaded instead of re-indexing everything
- [ ] Verified on a machine other than the author's

**Owner's call:** full Docker (with GPU via WSL2) or Docker for Qdrant only + a launch script.

---

## Epic D: Product & Story
`epic` `zone:product` · owner: **D**

**Goal.** Make the system easy to show and the project's story easy to follow: from the problem and existing solutions to the demo and the final presentation.

**Epic success criteria**
- The demo runs from a clean clone following the script
- Review of 3+ existing solutions
- The presentation covers all 5 required stages
- Materials in Google Drive, link in the course worksheet by 02.12
- Slides for stages "1. Problem statement" and "2. Review of existing solutions"

### D1. Problem statement and use cases
`zone:product` · **M1**

**Why.** Stage 1 of the presentation. It also gives the whole team a shared understanding of what we're building and for whom.

**Outcome**
- [ ] Problem statement, target user, 3–5 use cases
- [ ] A clear description of the big data aspect (volume, unstructured data, parallelism, vector index)
- [ ] Project scope: what we deliberately don't do

### D2. Review of existing solutions
`zone:product` · **M2**

**Why.** Stage 2 of the presentation: show that we know the landscape and how our solution is similar or different.

**Outcome**
- [ ] 3+ solutions or approaches: commercial products, open source, research papers
- [ ] A comparison table across several criteria
- [ ] A conclusion on what we took from them and why

### D3. Web UI
`zone:product` · **M2** · depends on: L1 (on fixtures first), then C3

**Why.** The face of the system in the demo. This is how the instructor will see that search works.

**Outcome**
- [ ] Query field, K selector, results with thumbnails, score and timestamps
- [ ] Playback of the found fragment
- [ ] Uploading a new video and showing its processing status
- [ ] Clear states: "nothing found", "API unavailable", "processing"

**Owner's call:** Gradio (recommended, mounts into FastAPI) or something else; design and layout.

### D4. Demo script
`zone:product` · **M3** · depends on: B5, D3

**Why.** There's no time to improvise at the defense. The demo has to reliably show the system's strengths.

**Outcome**
- [ ] 5–7 queries that showcase the system well, including one "wow" and one honest failure with an explanation
- [ ] A step-by-step script for 3–5 minutes
- [ ] A fallback in case of failure: a screen recording

### D5. Presentation and materials
`zone:product` · **M3** · each owner provides the slides for their zone

**Why.** The main artifact of the defense. Deadline 02.12.

**Outcome**
- [ ] One presentation: 5 stages + architecture + results + conclusions
- [ ] Slides from every zone collected and brought into one style
- [ ] Google Drive folder (presentation, repo link, demo recording), link in the course worksheet
- [ ] Defense rehearsal with everyone, timed

---

## Dependencies

```
L1 ─┬─> A2 ─> A3 ──────────┐
    ├─> B2 ─┬─> C2 ─> C3 ──┼─> D3 ─> C5
    │       └──────────────┼─> B3 ─> B4 ─> B5 ─> C4, D4
    └─> D3 (on fixtures)   ┘
A3 ─> A4
```
