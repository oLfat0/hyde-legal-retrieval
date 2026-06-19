"""
pca_vectors.py
--------------
Visualizacao PCA dos vetores intermediarios, com multiplos pares de componentes
(PC1xPC2, PC1xPC3, PC2xPC3, PC3xPC4) + scree plot de variancia acumulada.

Mostra mais da estrutura total do que apenas PC1xPC2 (que captura ~19% em
embeddings de 768-dim — limitacao esperada, nao defeito).

Le os vetores DIRETO das fontes originais (igual ao tsne_vectors.py):
  - f(ementa)    : extraido do indice FAISS
  - f(descricao) : data/vectors/query_descricoes.npy (ou computa on-the-fly)
  - f(mean HyDE) : HYDE_V2_EMBEDS_DIR/hyde_v2_embeds_mean.npy

Uso:
  python -m src.pca_vectors

Saida:
  data/vectors/pca_multi.csv   (PC1..PC4 para os 3 tipos)
  data/vectors/pca_multi.png   (4 paineis de pares + scree plot)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.assets_v2 import QUERIES_V2_PATH, FAISS_V2_DIR, HYDE_V2_EMBEDS_DIR

VECTORS_DIR  = Path("data/vectors")
FAISS_INDEX  = Path(FAISS_V2_DIR) / "corpus.index"
FAISS_META   = Path(FAISS_V2_DIR) / "corpus_meta.pkl"
HYDE_MEAN    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_mean.npy"
HYDE_META    = Path(HYDE_V2_EMBEDS_DIR) / "hyde_v2_embeds_meta.json"
QUERY_EMBEDS = VECTORS_DIR / "query_descricoes.npy"

COLORS = {
    "ementa": ("#1D9E75", "o", "f(ementa) — corpus (alvo)"),
    "query" : ("#378ADD", "^", "f(descricao) — baseline"),
    "hyde"  : ("#EF9F27", "s", "f(mean) — HyDE"),
}


def _load_corpus_embeddings() -> tuple[np.ndarray, list[str]]:
    print("[pca] Extraindo f(ementa) do indice FAISS...")
    index = faiss.read_index(str(FAISS_INDEX))
    with open(FAISS_META, "rb") as f:
        cdac_list = pickle.load(f)
    n, d = index.ntotal, index.d
    emb = np.zeros((n, d), dtype=np.float32)
    index.reconstruct_n(0, n, emb)
    return emb, cdac_list


def _load_hyde_mean() -> tuple[np.ndarray, list[str]]:
    print("[pca] Carregando f(mean HyDE)...")
    emb = np.load(HYDE_MEAN)
    with open(HYDE_META, "r", encoding="utf-8") as f:
        cdac_order = json.load(f)["cdacordao_order"]
    return emb, cdac_order


def _load_query_embeddings(cdac_order: list[str]) -> np.ndarray:
    if QUERY_EMBEDS.exists():
        print("[pca] f(descricao) carregando do disco...")
        return np.load(QUERY_EMBEDS)
    print("[pca] Computando f(descricao) on-the-fly...")
    from src.embedder_v2 import load_model, encode_texts
    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    query_map = {q["cdacordao"]: q["query"] for q in queries}
    texts = [query_map[cid] for cid in cdac_order]
    model, tokenizer = load_model()
    emb = encode_texts(texts, model, tokenizer, desc="Enc f(descricao)")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(QUERY_EMBEDS, emb)
    return emb


def _load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    emb_corpus, cdac_corpus = _load_corpus_embeddings()
    emb_hyde,   cdac_order  = _load_hyde_mean()
    emb_query               = _load_query_embeddings(cdac_order)
    corpus_pos = {cid: i for i, cid in enumerate(cdac_corpus)}
    idx = [corpus_pos[cid] for cid in cdac_order]
    return emb_corpus[idx], emb_query, emb_hyde, cdac_order


def run_pca(n_components: int = 6, seed: int = 42) -> None:
    emb_corpus, emb_query, emb_hyde, cdac_order = _load_all()
    n = len(cdac_order)

    all_emb = np.vstack([emb_corpus, emb_query, emb_hyde])
    print(f"[pca] Ajustando PCA ({n_components} componentes) sobre {all_emb.shape}...")

    pca = PCA(n_components=n_components, random_state=seed)
    all_pc = pca.fit_transform(all_emb)
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)

    ementa_pc = all_pc[0*n:1*n]
    query_pc  = all_pc[1*n:2*n]
    hyde_pc   = all_pc[2*n:3*n]
    groups = {"ementa": ementa_pc, "query": query_pc, "hyde": hyde_pc}

    # CSV
    rows = ["cdacordao,tipo," + ",".join(f"PC{i+1}" for i in range(n_components))]
    label_map = {"ementa": "f(ementa)", "query": "f(descricao)", "hyde": "f(mean HyDE)"}
    for key, pc in groups.items():
        for i, cid in enumerate(cdac_order):
            coords = ",".join(f"{pc[i,j]:.6f}" for j in range(n_components))
            rows.append(f"{cid},{label_map[key]},{coords}")
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORS_DIR / "pca_multi.csv", "w", encoding="utf-8-sig") as f:
        f.write("\n".join(rows))
    print("[pca] pca_multi.csv salvo")

    # Plot: 4 pares de PCs + 1 scree plot + 1 painel de texto = 2x3 grid
    pairs = [(0,1), (0,2), (1,2), (2,3)]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Espaco vetorial via PCA — multiplos componentes",
                 fontsize=14, fontweight="bold", y=0.99)
    axes = axes.flatten()

    for ax_i, (a, b) in enumerate(pairs):
        ax = axes[ax_i]
        for key, (color, marker, label) in COLORS.items():
            pc = groups[key]
            ax.scatter(pc[:,a], pc[:,b], c=color, marker=marker,
                       s=22, alpha=0.55, label=label, linewidths=0)
        ax.set_title(f"PC{a+1} x PC{b+1}  "
                     f"({(var[a]+var[b])*100:.1f}% da variancia)", fontsize=11)
        ax.set_xlabel(f"PC{a+1} ({var[a]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC{b+1} ({var[b]*100:.1f}%)", fontsize=9)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.spines[["top","right"]].set_visible(False)
        if ax_i == 0:
            ax.legend(fontsize=8, framealpha=0.85, loc="best")

    # Scree plot (variancia individual + acumulada)
    ax_scree = axes[4]
    pcs = np.arange(1, n_components + 1)
    ax_scree.bar(pcs, var * 100, color="#888780", alpha=0.7, label="individual")
    ax_scree.plot(pcs, cumvar * 100, color="#D85A30", marker="o",
                  linewidth=1.5, label="acumulada")
    for x, y in zip(pcs, cumvar * 100):
        ax_scree.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                          xytext=(0, 8), fontsize=8, ha="center", color="#993C1D")
    ax_scree.set_title("Variancia explicada por componente", fontsize=11)
    ax_scree.set_xlabel("Componente principal", fontsize=9)
    ax_scree.set_ylabel("% da variancia", fontsize=9)
    ax_scree.legend(fontsize=8, framealpha=0.85)
    ax_scree.grid(True, alpha=0.15, linewidth=0.5, axis="y")
    ax_scree.spines[["top","right"]].set_visible(False)

    # Painel de texto explicativo
    ax_txt = axes[5]
    ax_txt.axis("off")
    txt = (
        "Leitura:\n\n"
        f"• PC1+PC2 capturam {cumvar[1]*100:.1f}% da variancia.\n"
        f"• Os 4 primeiros PCs: {cumvar[3]*100:.1f}%.\n"
        f"• Os {n_components} primeiros: {cumvar[-1]*100:.1f}%.\n\n"
        "Valores 'baixos' sao esperados:\n"
        "embeddings de 768-dim distribuem\n"
        "significado por centenas de eixos.\n"
        "Nenhum par de PCs concentra tudo.\n\n"
        "Para distancias confiaveis use PCA;\n"
        "para clusters locais use t-SNE."
    )
    ax_txt.text(0.02, 0.97, txt, fontsize=10, va="top", ha="left",
                family="monospace", color="#2C2C2A",
                linespacing=1.5)

    plt.tight_layout()
    out_png = VECTORS_DIR / "pca_multi.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[pca] pca_multi.png salvo em {out_png}")

    print(f"\n[pca] === Variancia explicada ===")
    for i in range(n_components):
        print(f"  PC{i+1}: {var[i]*100:5.2f}%  (acumulada: {cumvar[i]*100:5.2f}%)")


if __name__ == "__main__":
    run_pca()