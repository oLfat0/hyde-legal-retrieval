"""
retriever_v2.py
---------------
BM25 e RRF para os experimentos v2.
Identico ao retriever.py, mas indexa o campo "ementa" do corpus v2.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.assets_v2 import CORPUS_V2_PATH, BM25_V2_DIR

BM25_INDEX_PATH = Path(BM25_V2_DIR) / "bm25.pkl"
BM25_META_PATH  = Path(BM25_V2_DIR) / "bm25_meta.pkl"
RRF_K           = 60


def _preprocess(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t and not t.isdigit()]
    return tokens


def build_bm25_index(force_rebuild: bool = False) -> tuple[BM25Okapi, list[str]]:
    """Constroi indice BM25 sobre o campo 'ementa' do corpus v2."""
    Path(BM25_V2_DIR).mkdir(parents=True, exist_ok=True)

    if not force_rebuild and BM25_INDEX_PATH.exists() and BM25_META_PATH.exists():
        print("[retriever_v2] Indice BM25 v2 encontrado — carregando...")
        return load_bm25_index()

    print("[retriever_v2] Construindo indice BM25 v2 (campo: ementa)...")

    with open(CORPUS_V2_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    cdacordao_list   = [doc["cdacordao"] for doc in corpus]
    # BM25 tambem indexa apenas a ementa
    tokenized_corpus = [
        _preprocess(doc["ementa"])
        for doc in tqdm(corpus, desc="Tokenizando ementas (BM25 v2)", unit="doc")
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_META_PATH, "wb") as f:
        pickle.dump(cdacordao_list, f)

    print(f"[retriever_v2] BM25 v2 salvo — {len(cdacordao_list)} ementas")
    return bm25, cdacordao_list


def load_bm25_index() -> tuple[BM25Okapi, list[str]]:
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_META_PATH, "rb") as f:
        cdacordao_list = pickle.load(f)
    print(f"[retriever_v2] BM25 v2 carregado — {len(cdacordao_list)} docs")
    return bm25, cdacordao_list


def sparse_search(
    query_texts: list[str],
    bm25: BM25Okapi,
    cdacordao_list: list[str],
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """BM25 sobre queries (descricao original ou ementa hipotetica — definido pelo chamador)."""
    rankings = []
    for query_text in tqdm(query_texts, desc="BM25 v2 retrieval", unit="query"):
        tokenized_query = _preprocess(query_text)
        scores      = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        ranking     = [(cdacordao_list[idx], float(scores[idx])) for idx in top_indices]
        rankings.append(ranking)
    return rankings


def reciprocal_rank_fusion(
    dense_rankings : list[list[tuple[str, float]]],
    sparse_rankings: list[list[tuple[str, float]]],
    k    : int = RRF_K,
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    assert len(dense_rankings) == len(sparse_rankings)
    hybrid_rankings = []
    for dense_rank, sparse_rank in zip(dense_rankings, sparse_rankings):
        rrf: dict[str, float] = {}
        for pos, (cid, _) in enumerate(dense_rank,  start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + pos)
        for pos, (cid, _) in enumerate(sparse_rank, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + pos)
        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hybrid_rankings.append(fused)
    return hybrid_rankings