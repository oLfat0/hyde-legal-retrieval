"""
export_vectors.py
-----------------
Exporta os vetores intermediarios do pipeline v2:

1. CSVs brutos (501 linhas x 768 colunas):
     data/vectors/csv/
       corpus_ementas.csv       <- f(ementa)
       query_descricoes.csv     <- f(descricao)
       hyde_mean.csv            <- f(mean)

2. Reducao PCA 2D para visualizacao:
     data/vectors/
       pca_2d.csv               <- x, y, tipo, cdacordao
       pca_2d.png               <- scatter plot salvo como imagem

=== Ordem de Execução ===
# Primeira vez — computa e salva f(descrição) + exporta tudo
python -m src.export_vectors --compute-queries

# Próximas vezes — f(descrição) já está em disco, só reexporta
python -m src.export_vectors
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.assets_v2 import (
    CORPUS_V2_PATH,
    QUERIES_V2_PATH,
    FAISS_V2_DIR,
    HYDE_V2_EMBEDS_DIR,
)

VECTORS_DIR  = Path("data/vectors")
CSV_DIR      = VECTORS_DIR / "csv"
FAISS_INDEX  = Path(FAISS_V2_DIR) / "corpus.index"
FAISS_META   = Path(FAISS_V2_DIR) / "corpus_meta.pkl"
HYDE_MEAN    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_mean.npy"
HYDE_META    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_meta.json"
QUERY_EMBEDS = VECTORS_DIR / "query_descricoes.npy"


# -- Extracao dos vetores ------------------------------------------------------

def extract_corpus_embeddings() -> tuple[np.ndarray, list[str]]:
    print("[export] Extraindo f(ementa) do indice FAISS...")
    index = faiss.read_index(str(FAISS_INDEX))
    with open(FAISS_META, "rb") as f:
        cdac_list = pickle.load(f)
    n, d = index.ntotal, index.d
    emb = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, emb)
    print(f"[export] f(ementa): {emb.shape}")
    return emb, cdac_list


def compute_or_load_query_embeddings(force: bool = False) -> tuple[np.ndarray, list[str]]:
    with open(HYDE_META, "r", encoding="utf-8") as f:
        cdac_order = json.load(f)["cdacordao_order"]

    if not force and QUERY_EMBEDS.exists():
        print("[export] f(descricao) encontrado em disco — carregando...")
        return np.load(QUERY_EMBEDS), cdac_order

    print("[export] Computando f(descricao)...")
    from src.embedder_v2 import load_model, encode_texts

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    query_map = {q["cdacordao"]: q["query"] for q in queries}
    texts = [query_map[cid] for cid in cdac_order]

    model, tokenizer = load_model()
    emb = encode_texts(texts, model, tokenizer, desc="Enc f(descricao)")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(QUERY_EMBEDS, emb)
    print(f"[export] f(descricao) salvo: {emb.shape}")
    return emb, cdac_order


def load_hyde_mean() -> tuple[np.ndarray, list[str]]:
    print("[export] Carregando f(mean)...")
    emb = np.load(HYDE_MEAN)
    with open(HYDE_META, "r", encoding="utf-8") as f:
        cdac_order = json.load(f)["cdacordao_order"]
    print(f"[export] f(mean): {emb.shape}")
    return emb, cdac_order


# -- Export CSV bruto ----------------------------------------------------------

def export_raw_csvs(
    emb_corpus:  np.ndarray,
    emb_query:   np.ndarray,
    emb_mean:    np.ndarray,
    cdac_order:  list[str],
    cdac_corpus: list[str],
):
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Alinha corpus na mesma ordem de cdac_order
    corpus_pos = {cid: i for i, cid in enumerate(cdac_corpus)}
    idx = [corpus_pos[cid] for cid in cdac_order]
    emb_corpus_aligned = emb_corpus[idx]

    dim = emb_corpus_aligned.shape[1]
    dim_cols = [f"dim_{i+1}" for i in range(dim)]

    for name, emb in [
        ("corpus_ementas",   emb_corpus_aligned),
        ("query_descricoes", emb_query),
        ("hyde_mean",        emb_mean),
    ]:
        df = pd.DataFrame(emb, columns=dim_cols)
        df.insert(0, "cdacordao", cdac_order)
        out = CSV_DIR / f"{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[export] {out.name} salvo — {df.shape[0]} linhas x {df.shape[1]} colunas")

    # Meta separado para facilitar leitura
    with open(CSV_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "cdacordao_order": cdac_order,
            "n_docs": len(cdac_order),
            "embedding_dim": dim,
            "descricao": {
                "corpus_ementas.csv":   "f(ementa)    — vetor de cada ementa indexada no FAISS",
                "query_descricoes.csv": "f(descricao) — vetor de cada descricao usada como query",
                "hyde_mean.csv":        "f(mean)      — media L2-normalizada das 5 rodadas HyDE",
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"[export] CSVs brutos salvos em {CSV_DIR}/")


# -- PCA 2D + visualizacao ----------------------------------------------------

def export_pca_plot(
    emb_corpus:  np.ndarray,
    emb_query:   np.ndarray,
    emb_mean:    np.ndarray,
    cdac_order:  list[str],
    cdac_corpus: list[str],
):
    print("[export] Reduzindo para 2D com PCA...")

    corpus_pos = {cid: i for i, cid in enumerate(cdac_corpus)}
    idx = [corpus_pos[cid] for cid in cdac_order]
    emb_corpus_aligned = emb_corpus[idx]

    # Empilha os 3 conjuntos para ajustar PCA no espaco combinado
    n = len(cdac_order)
    all_emb = np.vstack([emb_corpus_aligned, emb_query, emb_mean])  # (3N, 768)

    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_emb)
    var_explained = pca.explained_variance_ratio_

    ementa_2d = all_2d[0*n : 1*n]
    query_2d  = all_2d[1*n : 2*n]
    mean_2d   = all_2d[2*n : 3*n]

    # CSV da reducao 2D
    rows = []
    for i, cid in enumerate(cdac_order):
        rows.append({"cdacordao": cid, "tipo": "f(ementa)",    "x": ementa_2d[i,0], "y": ementa_2d[i,1]})
        rows.append({"cdacordao": cid, "tipo": "f(descricao)", "x": query_2d[i,0],  "y": query_2d[i,1]})
        rows.append({"cdacordao": cid, "tipo": "f(mean HyDE)", "x": mean_2d[i,0],   "y": mean_2d[i,1]})
    df_pca = pd.DataFrame(rows)
    df_pca.to_csv(VECTORS_DIR / "pca_2d.csv", index=False, encoding="utf-8-sig")
    print(f"[export] pca_2d.csv salvo — variancia explicada: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Espaco vetorial — PCA 2D (PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%)",
        fontsize=13, fontweight="bold", y=1.01
    )

    COLORS = {
        "f(ementa)"    : ("#1D9E75", "o", 40, 0.6, "f(ementa) — corpus"),
        "f(descricao)" : ("#378ADD", "^", 35, 0.5, "f(descricao) — baseline"),
        "f(mean HyDE)" : ("#EF9F27", "s", 35, 0.7, "f(mean) — HyDE"),
    }

    # Painel esquerdo: todos os pontos
    ax = axes[0]
    for tipo, (color, marker, size, alpha, label) in COLORS.items():
        d = df_pca[df_pca["tipo"] == tipo]
        ax.scatter(d["x"], d["y"], c=color, marker=marker,
                   s=size, alpha=alpha, label=label, linewidths=0)
    ax.set_title("Todos os documentos (N=501)", fontsize=11)
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.spines[["top","right"]].set_visible(False)

    # Painel direito: linhas conectando ementa <-> baseline <-> hyde por documento (amostra de 40)
    ax2 = axes[1]
    sample_idx = np.random.RandomState(42).choice(n, size=min(40, n), replace=False)

    for i in sample_idx:
        ex, ey = ementa_2d[i]
        qx, qy = query_2d[i]
        mx, my = mean_2d[i]
        # linha descricao -> ementa (baseline)
        ax2.plot([qx, ex], [qy, ey], color="#378ADD", alpha=0.25, linewidth=0.7)
        # linha hyde -> ementa
        ax2.plot([mx, ex], [my, ey], color="#EF9F27", alpha=0.35, linewidth=0.7)

    for tipo, (color, marker, size, alpha, label) in COLORS.items():
        d = df_pca[df_pca["tipo"] == tipo].iloc[sample_idx]
        ax2.scatter(d["x"], d["y"], c=color, marker=marker,
                    s=size*1.2, alpha=0.9, label=label, linewidths=0.5,
                    edgecolors="white")

    ax2.set_title("Amostra de 40 docs — linhas baseline (azul) e HyDE (laranja) → ementa", fontsize=10)
    ax2.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)", fontsize=10)
    ax2.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)", fontsize=10)
    ax2.legend(fontsize=9, framealpha=0.8)
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_png = VECTORS_DIR / "pca_2d.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[export] pca_2d.png salvo em {out_png}")


# -- Pipeline -----------------------------------------------------------------

def run(compute_queries: bool = False):
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    emb_corpus, cdac_corpus = extract_corpus_embeddings()
    emb_query,  cdac_order  = compute_or_load_query_embeddings(force=compute_queries)
    emb_mean,   _           = load_hyde_mean()

    export_raw_csvs(emb_corpus, emb_query, emb_mean, cdac_order, cdac_corpus)
    export_pca_plot(emb_corpus, emb_query, emb_mean, cdac_order, cdac_corpus)

    print("\n[export] Concluido. Arquivos em data/vectors/")


if __name__ == "__main__":
    run(compute_queries="--compute-queries" in sys.argv)