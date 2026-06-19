"""
hyde_embedder_rq3.py
--------------------
Computa embeddings e media para os 3 cenarios do RQ3.

Uso:
  python -m src.hyde_embedder_rq3 --granul curto
  python -m src.hyde_embedder_rq3 --granul medio
  python -m src.hyde_embedder_rq3 --granul longo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.assets_v2   import HYDE_N_ROUNDS
from src.embedder_v2 import encode_texts, load_model
from src.hyde_rq3    import load_round

EMBEDS_RQ3_DIR = Path("data/hyde_rq3_embeds")


def _embed_path(granul: str, n: int) -> Path:
    return EMBEDS_RQ3_DIR / granul / f"hyde_rq3_{granul}_{n:02d}.npy"


def _mean_path(granul: str) -> Path:
    return EMBEDS_RQ3_DIR / granul / "mean.npy"


def _meta_path(granul: str) -> Path:
    return EMBEDS_RQ3_DIR / granul / "meta.json"


def compute_round(granul: str, n: int, force: bool = False) -> np.ndarray:
    out = _embed_path(granul, n)
    if not force and out.exists():
        print(f"[hyde_embedder_rq3] {granul} rodada {n:02d}: carregando do disco")
        return np.load(out)

    print(f"[hyde_embedder_rq3] {granul} rodada {n:02d}: computando...")
    docs = load_round(granul, n)
    texts = [d["hyde_ementa"] for d in docs]
    cdacs = [d["cdacordao"]   for d in docs]

    if n == 1:
        _meta_path(granul).parent.mkdir(parents=True, exist_ok=True)
        with open(_meta_path(granul), "w", encoding="utf-8") as f:
            json.dump({"cdacordao_order": cdacs}, f, ensure_ascii=False, indent=2)

    model, tokenizer = load_model()
    emb = encode_texts(texts, model, tokenizer, desc=f"Enc RQ3 {granul} r{n:02d}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"[hyde_embedder_rq3] {granul} rodada {n:02d}: salvo {emb.shape}")
    return emb


def compute_mean(granul: str, force: bool = False) -> np.ndarray:
    mean_path = _mean_path(granul)
    if not force and mean_path.exists():
        print(f"[hyde_embedder_rq3] {granul} mean: carregando do disco")
        return np.load(mean_path)

    all_embs = [compute_round(granul, n) for n in range(1, HYDE_N_ROUNDS + 1)]
    stacked  = np.stack(all_embs, axis=0)
    mean_emb = stacked.mean(axis=0)
    norms    = np.linalg.norm(mean_emb, axis=1, keepdims=True).clip(min=1e-9)
    mean_emb = (mean_emb / norms).astype(np.float32)
    np.save(mean_path, mean_emb)
    print(f"[hyde_embedder_rq3] {granul} mean salvo: {mean_emb.shape}")
    return mean_emb


def run_pipeline(granul: str, force: bool = False) -> tuple[np.ndarray, list[str]]:
    """Retorna (mean_embeddings, cdacordao_order) para o cenario granul."""
    mean_emb = compute_mean(granul, force=force)
    with open(_meta_path(granul), "r", encoding="utf-8") as f:
        cdac_order = json.load(f)["cdacordao_order"]
    return mean_emb, cdac_order


if __name__ == "__main__":
    import time
    start = time.time()
    if start:
        print("Timer Iniciado...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--granul", choices=["curto","medio","longo"], required=True)
    args = parser.parse_args()
    mean_emb, order = run_pipeline(args.granul)
    end = time.time()
    print(f"Media final: shape={mean_emb.shape}, norm={np.linalg.norm(mean_emb, axis=1).mean():.4f}")
    print(f"Tempo: {(end-start)/60:.2f}min ({(end-start):.2f}s)")
