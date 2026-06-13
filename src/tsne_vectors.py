"""
tsne_vectors.py
---------------
Visualizacao t-SNE 2D dos vetores intermediarios do pipeline.

Reduz f(ementa), f(descricao) e f(mean HyDE) de 768 -> 2 dimensoes via t-SNE,
preservando estrutura local (clusters mais nitidos que o PCA).

Le os vetores DIRETO das fontes originais (nao depende de CSVs intermediarios):
  - f(ementa)    : extraido do indice FAISS (reconstruct_n)
  - f(descricao) : computado on-the-fly OU lido de data/vectors/query_descricoes.npy
  - f(mean HyDE) : lido de HYDE_V2_EMBEDS_DIR/hyde_v2_embeds_mean.npy

Uso:
  python -m src.tsne_vectors
  python -m src.tsne_vectors --perplexity 40 --sample 200

Saida (nao versionada — fica sob data/vectors/, ignorado pelo .gitignore):
  data/vectors/tsne_2d.csv
  data/vectors/tsne_2d.png
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.assets_v2 import (
    QUERIES_V2_PATH,
    FAISS_V2_DIR,
    HYDE_V2_EMBEDS_DIR,
)

VECTORS_DIR  = Path("data/vectors")
FAISS_INDEX  = Path(FAISS_V2_DIR) / "corpus.index"
FAISS_META   = Path(FAISS_V2_DIR) / "corpus_meta.pkl"
HYDE_MEAN    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_mean.npy"
HYDE_META    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_meta.json"
QUERY_EMBEDS = VECTORS_DIR / "query_descricoes.npy"


# -- Carregamento dos 3 conjuntos de vetores ----------------------------------

def _load_corpus_embeddings() -> tuple[np.ndarray, list[str]]:
    """f(ementa) — extrai do indice FAISS via reconstruct_n."""
    if not FAISS_INDEX.exists():
        raise FileNotFoundError(
            f"Indice FAISS nao encontrado: {FAISS_INDEX}\n"
            "Rode um experimento v2 antes (ex: baseline_dense_v2.py) para gera-lo."
        )
    print("[tsne] Extraindo f(ementa) do indice FAISS...")
    index = faiss.read_index(str(FAISS_INDEX))
    with open(FAISS_META, "rb") as f:
        cdac_list = pickle.load(f)
    n, d = index.ntotal, index.d
    emb = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, emb)
    print(f"[tsne] f(ementa): {emb.shape}")
    return emb, cdac_list


def _load_hyde_mean() -> tuple[np.ndarray, list[str]]:
    """f(mean HyDE) — le do diretorio de embeddings HyDE."""
    if not HYDE_MEAN.exists():
        raise FileNotFoundError(
            f"Media HyDE nao encontrada: {HYDE_MEAN}\n"
            "Rode src.hyde_embedder_v2 antes."
        )
    print("[tsne] Carregando f(mean HyDE)...")
    emb = np.load(HYDE_MEAN)
    with open(HYDE_META, "r", encoding="utf-8") as f:
        cdac_order = json.load(f)["cdacordao_order"]
    print(f"[tsne] f(mean HyDE): {emb.shape}")
    return emb, cdac_order


def _load_query_embeddings(cdac_order: list[str], force: bool = False) -> np.ndarray:
    """
    f(descricao) — le de data/vectors/query_descricoes.npy se existir,
    senao computa on-the-fly (mesma logica do export_vectors).
    """
    if not force and QUERY_EMBEDS.exists():
        print("[tsne] f(descricao) encontrado em disco — carregando...")
        return np.load(QUERY_EMBEDS)

    print("[tsne] Computando f(descricao) on-the-fly...")
    from src.embedder_v2 import load_model, encode_texts

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    query_map = {q["cdacordao"]: q["query"] for q in queries}
    texts = [query_map[cid] for cid in cdac_order]

    model, tokenizer = load_model()
    emb = encode_texts(texts, model, tokenizer, desc="Enc f(descricao)")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(QUERY_EMBEDS, emb)
    print(f"[tsne] f(descricao) salvo: {emb.shape}")
    return emb


def _load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Carrega os 3 conjuntos alinhados pela ordem do HyDE meta."""
    emb_corpus, cdac_corpus = _load_corpus_embeddings()
    emb_hyde,   cdac_order  = _load_hyde_mean()
    emb_query               = _load_query_embeddings(cdac_order)

    # Alinha o corpus na mesma ordem do cdac_order (ordem do HyDE)
    corpus_pos = {cid: i for i, cid in enumerate(cdac_corpus)}
    idx = [corpus_pos[cid] for cid in cdac_order]
    emb_corpus_aligned = emb_corpus[idx]

    return emb_corpus_aligned, emb_query, emb_hyde, cdac_order


# -- t-SNE + visualizacao -----------------------------------------------------

def run_tsne(perplexity: float = 30.0, sample: int | None = None, seed: int = 42) -> None:
    emb_corpus, emb_query, emb_hyde, cdac_order = _load_all()
    n_total = emb_corpus.shape[0]

    if sample and sample < n_total:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=sample, replace=False)
        emb_corpus = emb_corpus[idx]
        emb_query  = emb_query[idx]
        emb_hyde   = emb_hyde[idx]
        cdac_order = [cdac_order[i] for i in idx]
        n = sample
        print(f"[tsne] amostra de {sample} documentos (de {n_total})")
    else:
        n = n_total
        print(f"[tsne] usando todos os {n} documentos")

    if perplexity >= n:
        perplexity = max(5, n // 4)
        print(f"[tsne] perplexity ajustada para {perplexity} (limite por N)")

    all_emb = np.vstack([emb_corpus, emb_query, emb_hyde])  # (3n, 768)
    print(f"[tsne] reduzindo {all_emb.shape} -> 2D (perplexity={perplexity})...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
        metric="cosine",
    )
    all_2d = tsne.fit_transform(all_emb)

    ementa_2d = all_2d[0*n : 1*n]
    query_2d  = all_2d[1*n : 2*n]
    hyde_2d   = all_2d[2*n : 3*n]

    # CSV
    rows = ["cdacordao,tipo,x,y"]
    for i, cid in enumerate(cdac_order):
        rows.append(f"{cid},f(ementa),{ementa_2d[i,0]:.6f},{ementa_2d[i,1]:.6f}")
        rows.append(f"{cid},f(descricao),{query_2d[i,0]:.6f},{query_2d[i,1]:.6f}")
        rows.append(f"{cid},f(mean HyDE),{hyde_2d[i,0]:.6f},{hyde_2d[i,1]:.6f}")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORS_DIR / "tsne_2d.csv", "w", encoding="utf-8-sig") as f:
        f.write("\n".join(rows))
    print("[tsne] tsne_2d.csv salvo")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f"Espaco vetorial via t-SNE (perplexity={perplexity:.0f}, N={n})",
                 fontsize=13, fontweight="bold", y=1.0)

    ax = axes[0]
    ax.scatter(ementa_2d[:,0], ementa_2d[:,1], c="#1D9E75", marker="o",
               s=45, alpha=0.65, label="f(ementa) — corpus (alvo)", linewidths=0)
    ax.scatter(query_2d[:,0],  query_2d[:,1],  c="#378ADD", marker="^",
               s=40, alpha=0.55, label="f(descricao) — baseline", linewidths=0)
    ax.scatter(hyde_2d[:,0],   hyde_2d[:,1],   c="#EF9F27", marker="s",
               s=40, alpha=0.70, label="f(mean) — HyDE", linewidths=0)
    ax.set_title("Distribuicao dos 3 tipos de vetor", fontsize=11)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85, loc="best")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    ax2 = axes[1]
    n_lines = min(35, n)
    sample_idx = np.random.RandomState(seed).choice(n, size=n_lines, replace=False)
    for i in sample_idx:
        ex, ey = ementa_2d[i]; qx, qy = query_2d[i]; hx, hy = hyde_2d[i]
        ax2.plot([qx, ex], [qy, ey], color="#378ADD", alpha=0.30, linewidth=0.8)
        ax2.plot([hx, ex], [hy, ey], color="#EF9F27", alpha=0.45, linewidth=0.8)
    ax2.scatter(ementa_2d[sample_idx,0], ementa_2d[sample_idx,1], c="#1D9E75",
                marker="o", s=55, alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
    ax2.scatter(query_2d[sample_idx,0], query_2d[sample_idx,1], c="#378ADD",
                marker="^", s=48, alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
    ax2.scatter(hyde_2d[sample_idx,0], hyde_2d[sample_idx,1], c="#EF9F27",
                marker="s", s=48, alpha=0.9, edgecolors="white", linewidths=0.6, zorder=3)
    ax2.set_title(f"Conexoes ate a ementa-alvo (amostra de {n_lines})\n"
                  f"azul: descricao->ementa | laranja: HyDE->ementa", fontsize=10)
    ax2.set_xlabel("t-SNE dim 1", fontsize=10)
    ax2.set_ylabel("t-SNE dim 2", fontsize=10)
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_png = VECTORS_DIR / "tsne_2d.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[tsne] tsne_2d.png salvo em {out_png}")

    d_query = np.linalg.norm(query_2d - ementa_2d, axis=1).mean()
    d_hyde  = np.linalg.norm(hyde_2d  - ementa_2d, axis=1).mean()
    print(f"\n[tsne] === Distancia media ate a ementa-alvo (espaco t-SNE 2D) ===")
    print(f"  descricao -> ementa : {d_query:.3f}")
    print(f"  HyDE      -> ementa : {d_hyde:.3f}")
    if d_hyde < d_query:
        print(f"  -> HyDE fica {(1 - d_hyde/d_query)*100:.1f}% mais proximo (qualitativo)")
    print("  (nota: distancias t-SNE sao qualitativas, nao metricas absolutas)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_tsne(perplexity=args.perplexity, sample=args.sample, seed=args.seed)