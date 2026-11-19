"""Check that the environment works (B1).

Prints versions and device, embeds one image and one text, and checks they land
in the same space. Exit code 0 = OK.

Run: python scripts/check_env.py [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="override SVS_DEVICE")
    args = parser.parse_args()

    import open_clip
    import torch

    from svs.config import Settings, get_settings
    from svs.embeddings import get_embedder

    settings = get_settings()
    if args.device:
        settings = Settings(**{**settings.model_dump(), "device": args.device})

    print(f"python      {platform.python_version()} ({platform.system()})")
    print(f"torch       {torch.__version__}")
    print(f"open_clip   {open_clip.__version__}")
    print(f"cuda        {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" | {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}", end="")
    print()
    print(f"model       {settings.model_tag}")

    t0 = time.perf_counter()
    emb = get_embedder(settings)
    print(f"device      {emb.device}")
    print(f"dim         {emb.dim}")
    print(f"load        {time.perf_counter() - t0:.1f}s")

    red = Image.new("RGB", (224, 224), (220, 30, 30))
    t0 = time.perf_counter()
    img = emb.embed_images([red])
    txt = emb.embed_text(["a red image", "a blue image"])
    print(f"embed       {(time.perf_counter() - t0) * 1000:.0f} ms (1 image + 2 texts)")

    sims = (txt @ img.T).ravel()
    print(f"sim(red img, 'a red image')  = {sims[0]:.3f}")
    print(f"sim(red img, 'a blue image') = {sims[1]:.3f}")

    ok = (
        img.shape == (1, emb.dim)
        and np.allclose(np.linalg.norm(img, axis=1), 1, atol=1e-3)
        and sims[0] > sims[1]
    )
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
