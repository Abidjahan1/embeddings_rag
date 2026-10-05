"""
Encode Bengali passages using multilingual-e5 or BGE-M3.

Inputs:  --corpus-size (int), --output (path), --batch-size (int), --model-name (str)
Outputs: .npy file with shape (N, D) float32 embeddings
Verification: prints shape, dtype, first vector norm; saves a .json metadata file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-size", type=int, default=100,
                   help="Number of passages to encode (10k/25k/50k/100k in full runs)")
    p.add_argument("--output", type=str, required=True,
                   help="Path to save the .npy embeddings file")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-name", type=str,
                   default="intfloat/multilingual-e5-small",
                   help="SentenceTransformer model. e5-small first for a fast smoke test.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_device() -> str:
    """Return 'cuda' if available, else 'cpu'. Platform-agnostic."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_passages(corpus_size: int, seed: int) -> tuple[list[str], list[str]]:
    """
    Load MIRACL Bengali corpus and sample `corpus_size` passages.

    NOTE: This is a temporary sampling strategy for the smoke test.
    The proper nested-corpus sampler (that guarantees all judged-relevant
    passages are included) comes in a later module.

    Returns: (doc_ids, texts)
    """
    ds = load_dataset("miracl/miracl-corpus", "bn", trust_remote_code=True)
    split = ds["train"]  # verify split name; MIRACL corpus usually has 'train'
    split = split.shuffle(seed=seed).select(range(min(corpus_size, len(split))))
    doc_ids = [str(x["docid"]) for x in split]
    texts = [x["text"] for x in split]
    return doc_ids, texts


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"[info] device = {device}")
    if device == "cuda":
        print(f"[info] gpu = {torch.cuda.get_device_name(0)}")

    print(f"[info] loading MIRACL bn corpus, size = {args.corpus_size}")
    doc_ids, texts = load_passages(args.corpus_size, args.seed)

    print(f"[info] loading model {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)

    # E5 requires the "passage: " prefix. If you switch to BGE-M3,
    # remove the prefix (BGE-M3 does not use prefixes).
    use_prefix = "e5" in args.model_name.lower()
    prefixed = [f"passage: {t}" for t in texts] if use_prefix else texts

    print(f"[info] encoding {len(prefixed)} passages, batch_size={args.batch_size}")
    embeddings = model.encode(
        prefixed,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity == dot product
    ).astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"[info] saved embeddings to {out_path}, shape = {embeddings.shape}")

    # Metadata for reproducibility
    meta = {
        "model_name": args.model_name,
        "corpus_size": args.corpus_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "use_prefix": use_prefix,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "doc_ids_head": doc_ids[:5],
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[info] saved metadata to {meta_path}")


if __name__ == "__main__":
    main()