# Decisions

Short log: what was decided and why. Newest at the bottom.

## Zone B / L1 (contract)

- **IDs.** `video_id` = sha1 of the file name (16 hex); `segment_id` = UUIDv5 of
  `video_id:index`. Why: Qdrant point ids must be int or UUID; deterministic ids make re-indexing
  idempotent and give the same ids on every machine.
- **Vectors per segment vs per frame.** Both supported (`SVS_VECTOR_MODE=segment|frame`), default
  `segment` (mean of frame vectors). Why: segment mode is smaller and simpler; frame mode may find
  short moments better. B4 will compare them with numbers.
- **Model marking.** One Qdrant collection per model + mode, plus a `model` payload field.
  Why: different models have different vector sizes, so they can't share one collection.
- **Normalized vectors, cosine distance.** Why: dot product = cosine, works the same in Qdrant,
  numpy and FAISS.
- **Library: open_clip.** Why: CLIP and SigLIP behind one API, model switched by two config
  strings. Default `ViT-B-32/laion2b_s34b_b79k`: small (512-d), fast on CPU, good quality.
- **fp16 on GPU only** (autocast), fp32 on CPU. Why: speed on GPU, CPU fp16 is slow/unsupported.
- **Config via pydantic-settings**, prefix `SVS_`, optional `.env`. Why: typed, validated, one place.
