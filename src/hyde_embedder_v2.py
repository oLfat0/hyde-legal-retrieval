"""
hyde_embedder_v2.py
-------------------
Computa embeddings das ementas hipotetrias (HyDE v2) e calcula a media.

Estrutura gerada:
  data/hyde_v2_embeds/
    hyde_v2_embeds_01.npy   shape: (N, D)
    ...
    hyde_v2_embeds_05.npy
    hyde_v2_embeds_mean.npy shape: (N, D) — media L2-normalizada
    hyde_v2_embeds_meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import hyde_v2 as hyde_module
from src.assets_v2 import HYDE_V2_EMBEDS_DIR, HYDE_N_ROUNDS
from src.embedder_v2 import encode_texts, load_model

EMBEDS_DIR  = Path(HYDE_V2_EMBEDS_DIR)
MEAN_PATH   = EMBEDS_DIR / "hyde_v2_embeds_mean.npy"
META_PATH   = EMBEDS_DIR / "hyde_v2_embeds_meta.json"


def _embed_path(n: int) -> Path:
    return EMBEDS_DIR / f"hyde_v2_embeds_{n:02d}.npy"


def compute_round_embeddings(n: int, force_recompute: bool = False) -> np.ndarray:
    EMBEDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _embed_path(n)

    if not force_recompute and out_path.exists():
        print(f"[hyde_embedder_v2] rodada {n:02d}: carregando do disco")
        return np.load(out_path)

    print(f"[hyde_embedder_v2] rodada {n:02d}: computando embeddings...")
    docs = hyde_module.load_round(n)

    # Embeda a ementa hipotetica gerada (hyde_ementa), nao a descricao
    hyde_texts     = [d["hyde_ementa"] for d in docs]
    cdacordao_list = [d["cdacordao"]   for d in docs]

    if n == 1:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"cdacordao_order": cdacordao_list}, f, ensure_ascii=False, indent=2)

    model, tokenizer = load_model()
    embeddings = encode_texts(hyde_texts, model, tokenizer, desc=f"Enc rodada {n:02d}")

    np.save(out_path, embeddings)
    print(f"[hyde_embedder_v2] rodada {n:02d}: salvo {embeddings.shape}")
    return embeddings


def compute_all_round_embeddings(force_recompute: bool = False) -> list[np.ndarray]:
    return [
        compute_round_embeddings(n, force_recompute=force_recompute)
        for n in range(1, HYDE_N_ROUNDS + 1)
    ]


def compute_mean_embeddings(
    all_embeddings: list[np.ndarray] | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Media L2-normalizada dos embeddings das 5 rodadas.
    v_mean = normalize( mean(f(h1), ..., f(h5)) )
    """
    EMBEDS_DIR.mkdir(parents=True, exist_ok=True)

    if not force_recompute and MEAN_PATH.exists():
        print(f"[hyde_embedder_v2] media em disco — carregando")
        return np.load(MEAN_PATH)

    if all_embeddings is None:
        all_embeddings = [np.load(_embed_path(n)) for n in range(1, HYDE_N_ROUNDS + 1)]

    stacked  = np.stack(all_embeddings, axis=0)   # (R, N, D)
    mean_emb = stacked.mean(axis=0)               # (N, D)
    norms    = np.linalg.norm(mean_emb, axis=1, keepdims=True).clip(min=1e-9)
    mean_emb = (mean_emb / norms).astype(np.float32)

    np.save(MEAN_PATH, mean_emb)
    print(f"[hyde_embedder_v2] media salva: {mean_emb.shape}")
    return mean_emb


def load_meta() -> list[str]:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta nao encontrado: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cdacordao_order"]


def run_full_pipeline(force_recompute: bool = False) -> tuple[np.ndarray, list[str]]:
    """Pipeline completo: computa todas as rodadas + media. Retorna (mean_emb, cdacordao_order)."""
    all_embeddings  = compute_all_round_embeddings(force_recompute=force_recompute)
    mean_embeddings = compute_mean_embeddings(all_embeddings, force_recompute=force_recompute)
    cdacordao_order = load_meta()
    return mean_embeddings, cdacordao_order


if __name__ == "__main__":
    mean_emb, order = run_full_pipeline()
    print(f"Media final: shape={mean_emb.shape}, norm_media={np.linalg.norm(mean_emb, axis=1).mean():.4f}")