# Semantic Video Search

Find moments inside videos with a text query ("a dog running on the beach").
Videos are cut into segments, frames are embedded with CLIP-like models,
vectors go to Qdrant, and search returns segments with timestamps.

- Backlog and team agreements: [`docs/backlog.md`](docs/backlog.md)
- Data contract between zones: [`docs/interfaces.md`](docs/interfaces.md)
- Decisions log: [`docs/decisions.md`](docs/decisions.md)

## Install

Python 3.10+. Commands below use [uv](https://docs.astral.sh/uv/); plain `pip` works the same.

### CPU (most of the team)

```bash
uv venv -p 3.12
uv pip install -e ".[ml,dev]"
```

### GPU (NVIDIA, incl. RTX 50-series)

RTX 50-series (Blackwell) needs PyTorch built for **CUDA 12.8 or newer**. Install torch from the
PyTorch index first, then the project:

```bash
uv venv -p 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[ml,dev]"
```

Your NVIDIA driver must support CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.8" or higher).
On Windows this works natively; WSL2 also works.

### Check the environment

```bash
python scripts/check_env.py            # uses SVS_DEVICE (default: auto)
python scripts/check_env.py --device cuda
```

Prints versions and device, embeds one image and two texts, and ends with `OK`.
The first run downloads the model (~600 MB) into the Hugging Face cache.

## Configuration

All parameters live in [`src/svs/config.py`](src/svs/config.py) and are set via env vars
`SVS_<NAME>` or a `.env` file. Start from `.env.example`.

## Tests

```bash
pytest -m "not slow"   # fast, no model download (use this in CI)
pytest                 # also runs real-model tests on CPU (~30 s after the download)
ruff check . && ruff format --check .
```

Regenerate test fixtures: `python scripts/make_fixtures.py`.
