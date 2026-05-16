"""
hyde_embedder.py
----------------
Responsável por:
  - Computar embeddings de cada rodada de hyde_docs
  - Salvar embeddings em data/hyde_embeds/hyde_embeds_{n:02d}.npy
  - Computar e salvar a média dos N embeddings por query
  - Fornecer o array médio pronto para uso nos experimentos

Estrutura de arquivos gerada:
  data/hyde_embeds/
    hyde_embeds_01.npy   shape: (N_queries, D)
    hyde_embeds_02.npy
    ...
    hyde_embeds_10.npy
    hyde_embeds_mean.npy shape: (N_queries, D) — média L2-normalizada
    hyde_embeds_meta.json — ordem dos cdacordao (posição -> cdacordao)

A média é calculada sobre os vetores brutos e depois renormalizada (L2),
o que é matematicamente equivalente ao centroide normalizado na hiperesfera.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import hyde as hyde_module
from src.embedder import encode_texts, load_model

# -- Configuracoes -------------------------------------------------------------
HYDE_EMBEDS_DIR  = Path("data/hyde_embeds")
MEAN_EMBEDS_PATH = HYDE_EMBEDS_DIR / "hyde_embeds_mean.npy"
META_PATH        = HYDE_EMBEDS_DIR / "hyde_embeds_meta.json"


def _embed_path(n: int) -> Path:
    return HYDE_EMBEDS_DIR / f"hyde_embeds_{n:02d}.npy"


# -- Geracao de embeddings por rodada ------------------------------------------

def compute_round_embeddings(
    n: int,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Computa e persiste embeddings da rodada n.

    Args:
        n               : numero da rodada (1-based)
        force_recompute : recomputa mesmo se .npy ja existir

    Retorna array float32 (N_queries, D) L2-normalizado.
    """
    HYDE_EMBEDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _embed_path(n)

    if not force_recompute and out_path.exists():
        print(f"[hyde_embedder] rodada {n:02d}: embeddings em disco — carregando")
        return np.load(out_path)

    print(f"[hyde_embedder] rodada {n:02d}: computando embeddings...")
    docs = hyde_module.load_round(n)

    hyde_texts     = [d["hyde_doc"]  for d in docs]
    cdacordao_list = [d["cdacordao"] for d in docs]

    # Salva meta apenas na primeira rodada (ordem e identica em todas)
    if n == 1:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"cdacordao_order": cdacordao_list}, f, ensure_ascii=False, indent=2)

    model, tokenizer = load_model()
    embeddings = encode_texts(
        hyde_texts, model, tokenizer, desc=f"Encoding rodada {n:02d}"
    )

    np.save(out_path, embeddings)
    print(f"[hyde_embedder] rodada {n:02d}: salvo {embeddings.shape} em {out_path}")
    return embeddings


def compute_all_round_embeddings(force_recompute: bool = False) -> list[np.ndarray]:
    """
    Computa embeddings de todas as rodadas.
    Retorna lista de N_ROUNDS arrays (N_queries, D).
    """
    n_rounds = hyde_module.N_ROUNDS
    all_embeddings = []
    for n in range(1, n_rounds + 1):
        emb = compute_round_embeddings(n, force_recompute=force_recompute)
        all_embeddings.append(emb)
    return all_embeddings


# -- Media dos embeddings ------------------------------------------------------

def compute_mean_embeddings(
    all_embeddings: list[np.ndarray] | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Calcula a media L2-normalizada dos embeddings de todas as rodadas.

    Matematicamente:
        v_mean = normalize( (1/N) * sum(f(h^1), f(h^2), ..., f(h^N)) )

    Normalizar apos a media e correto: o centroide de vetores unitarios
    nao e necessariamente unitario, entao renormalizamos para manter
    consistencia com o IndexFlatIP do FAISS (que assume norma 1).

    Args:
        all_embeddings  : lista de arrays ja computados (evita recarregar do disco)
        force_recompute : recomputa mesmo se mean.npy ja existir

    Retorna array float32 (N_queries, D) L2-normalizado.
    """
    HYDE_EMBEDS_DIR.mkdir(parents=True, exist_ok=True)

    if not force_recompute and MEAN_EMBEDS_PATH.exists():
        print(f"[hyde_embedder] media em disco — carregando {MEAN_EMBEDS_PATH}")
        return np.load(MEAN_EMBEDS_PATH)

    if all_embeddings is None:
        print("[hyde_embedder] Carregando embeddings de todas as rodadas do disco...")
        all_embeddings = _load_all_round_embeddings()

    n_rounds = len(all_embeddings)
    print(f"[hyde_embedder] Calculando media de {n_rounds} rodadas...")

    # Stack: (N_rounds, N_queries, D) -> mean ao longo do eixo 0
    stacked = np.stack(all_embeddings, axis=0)         # (R, N, D)
    mean_emb = stacked.mean(axis=0)                    # (N, D)

    # Renormalizacao L2 apos averaging
    norms = np.linalg.norm(mean_emb, axis=1, keepdims=True).clip(min=1e-9)
    mean_emb = (mean_emb / norms).astype(np.float32)   # (N, D)

    np.save(MEAN_EMBEDS_PATH, mean_emb)
    print(f"[hyde_embedder] media salva: {mean_emb.shape} em {MEAN_EMBEDS_PATH}")
    return mean_emb


def _load_all_round_embeddings() -> list[np.ndarray]:
    """Carrega todos os .npy de rodadas do disco."""
    n_rounds = hyde_module.N_ROUNDS
    all_embs = []
    for n in range(1, n_rounds + 1):
        path = _embed_path(n)
        if not path.exists():
            raise FileNotFoundError(
                f"Embeddings da rodada {n:02d} nao encontrados: {path}\n"
                f"Execute: python -m src.hyde_embedder"
            )
        all_embs.append(np.load(path))
    return all_embs


# -- Carregamento do meta (cdacordao_order) ------------------------------------

def load_meta() -> list[str]:
    """Retorna a lista ordenada de cdacordao (posicao -> cdacordao)."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Meta nao encontrado: {META_PATH}\n"
            "Execute compute_all_round_embeddings() primeiro."
        )
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cdacordao_order"]


# -- Pipeline completo ---------------------------------------------------------

def run_full_pipeline(force_recompute: bool = False) -> tuple[np.ndarray, list[str]]:
    """
    Executa o pipeline completo:
      1. Computa embeddings de cada rodada (com cache)
      2. Calcula a media L2-normalizada
      3. Retorna (mean_embeddings, cdacordao_order)

    Use este metodo nos experimentos hyde_dense.py e hyde_hybrid.py.
    """
    all_embeddings = compute_all_round_embeddings(force_recompute=force_recompute)
    mean_embeddings = compute_mean_embeddings(all_embeddings, force_recompute=force_recompute)
    cdacordao_order = load_meta()
    return mean_embeddings, cdacordao_order


if __name__ == "__main__":
    import time

    print("=== Pipeline de embeddings HyDE (multi-round averaging) ===")
    print("[hyde_embedder] Timer iniciado...")
    start = time.time()
    mean_emb, order = run_full_pipeline()
    print(f"Media final: shape={mean_emb.shape}, norm_media={np.linalg.norm(mean_emb, axis=1).mean():.4f}")
    print(f"Ordem: {len(order)} documentos")
    end = time.time()
    tempo = end-start
    print(f"[hyde_embedder] Tempo de Demora: {tempo/3600.00:.2f}h ({tempo/60.00:.2f}min | {tempo:.2f}s)")