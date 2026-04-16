"""
retriever.py
------------
Recuperacao hibrida: BM25 (lexical) + Dense (semantico) com fusao RRF.

Arquitetura da recuperacao hibrida conforme metodologia:
  - Dense  : usa embedding da query (baseline) ou do doc hipotetico (HyDE)
  - Sparse : BM25Okapi aplicado sobre a query ORIGINAL qi (nunca sobre hi)
  - Fusao  : Reciprocal Rank Fusion (RRF) — zero-shot, sem parametros treinados

Importante: o BM25 SEMPRE usa a query original qi, mesmo no experimento HyDE+Hybrid.
Isso está correto e intencional — preserva a precisão lexical da query real
enquanto o lado denso se beneficia da expansão semântica do HyDE.
"""

from __future__ import annotations

import json
import pickle
import re
import string
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.assets import CORPUS_PATH

# -- Configuracoes BM25 --------------------------------------------------------
BM25_INDEX_DIR  = Path("data/bm25_index")
BM25_INDEX_PATH = BM25_INDEX_DIR / "bm25.pkl"
BM25_META_PATH  = BM25_INDEX_DIR / "bm25_meta.pkl"   # posicao -> cdacordao

# -- RRF -----------------------------------------------------------------------
RRF_K = 60   # constante padrao da literatura (Robertson et al.)


# -- Pre-processamento textual -------------------------------------------------

def _preprocess(text: str) -> list[str]:
    """
    Tokenizacao simples adequada para português juridico:
      - lowercase
      - remove pontuação e numeros isolados
      - split por espacos

    Nao aplicamos stemming para preservar termos juridicos especificos
    (ex: 'agravo', 'provimento', 'recursal') que seriam distorcidos por stemmers
    genericos de portugues.
    """
    text = text.lower()
    # Remove pontuacao mas preserva hifens internos (ex: "art.5o" -> "art5o")
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove tokens que sao apenas numeros (numeros de processo, artigos)
    tokens = [t for t in text.split() if t and not t.isdigit()]
    return tokens


# -- Construcao e persistencia do indice BM25 ----------------------------------

def build_bm25_index(force_rebuild: bool = False) -> tuple[BM25Okapi, list[str]]:
    """
    Constroi e persiste o indice BM25 sobre o corpus de ementas.

    Retorna:
        bm25           : instancia BM25Okapi indexada
        cdacordao_list : lista mapeando posicao -> cdacordao

    Cache: salvo em data/bm25_index/ e reutilizado nas execucoes seguintes.
    """
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and BM25_INDEX_PATH.exists() and BM25_META_PATH.exists():
        print("[retriever] Indice BM25 encontrado em disco - carregando...")
        return load_bm25_index()

    print("[retriever] Construindo indice BM25 a partir do corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    cdacordao_list  = [doc["cdacordao"] for doc in corpus]
    tokenized_corpus = [_preprocess(doc["texto"]) for doc in
                        tqdm(corpus, desc="Tokenizando corpus (BM25)", unit="doc")]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_META_PATH, "wb") as f:
        pickle.dump(cdacordao_list, f)

    print(f"[retriever] Indice BM25 salvo - {len(cdacordao_list)} docs")
    return bm25, cdacordao_list


def load_bm25_index() -> tuple[BM25Okapi, list[str]]:
    """Carrega indice BM25 e mapeamento do disco."""
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_META_PATH, "rb") as f:
        cdacordao_list = pickle.load(f)
    print(f"[retriever] Indice BM25 carregado - {len(cdacordao_list)} docs")
    return bm25, cdacordao_list


# -- Busca esparsa (BM25) ------------------------------------------------------

def sparse_search(
    query_texts: list[str],
    bm25: BM25Okapi,
    cdacordao_list: list[str],
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Busca lexical BM25 para N queries.

    IMPORTANTE: query_texts deve conter SEMPRE as queries originais qi,
    nunca os documentos hipotetricos hi.

    Retorna:
        Lista de N rankings: [(cdacordao, score), ...] decrescente por score BM25.
    """
    rankings: list[list[tuple[str, float]]] = []

    for query_text in tqdm(query_texts, desc="BM25 retrieval", unit="query"):
        tokenized_query = _preprocess(query_text)
        scores = bm25.get_scores(tokenized_query)      # array de shape (N_corpus,)

        # Pega indices dos top_k scores maiores
        top_indices = np.argsort(scores)[::-1][:top_k]

        ranking = [
            (cdacordao_list[idx], float(scores[idx]))
            for idx in top_indices
        ]
        rankings.append(ranking)

    return rankings


# -- Reciprocal Rank Fusion (RRF) ----------------------------------------------

def reciprocal_rank_fusion(
    dense_rankings: list[list[tuple[str, float]]],
    sparse_rankings: list[list[tuple[str, float]]],
    k: int = RRF_K,
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Funde rankings denso e esparso via RRF para N queries.

    Formula RRF para documento d na query i:
        RRF_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
    
    Documentos que aparecem em apenas um dos rankings recebem apenas
    a contribuicao daquele ranking (o outro termo e simplesmente omitido).

    Args:
        dense_rankings  : rankings do dense retrieval (baseline ou HyDE)
        sparse_rankings : rankings do BM25 (sempre com qi original)
        k               : constante RRF (default=60, padrao da literatura)
        top_k           : tamanho do ranking final retornado

    Retorna:
        Lista de N rankings fundidos: [(cdacordao, rrf_score), ...] desc.
    """
    assert len(dense_rankings) == len(sparse_rankings), (
        "dense_rankings e sparse_rankings devem ter o mesmo numero de queries"
    )

    hybrid_rankings: list[list[tuple[str, float]]] = []

    for dense_rank, sparse_rank in zip(dense_rankings, sparse_rankings):
        rrf_scores: dict[str, float] = {}

        # Contribuicao do ranking denso
        for pos, (cdacordao, _) in enumerate(dense_rank, start=1):
            rrf_scores[cdacordao] = rrf_scores.get(cdacordao, 0.0) + 1.0 / (k + pos)

        # Contribuicao do ranking esparso
        for pos, (cdacordao, _) in enumerate(sparse_rank, start=1):
            rrf_scores[cdacordao] = rrf_scores.get(cdacordao, 0.0) + 1.0 / (k + pos)

        # Ordena por score RRF decrescente e trunca em top_k
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        hybrid_rankings.append(fused)

    return hybrid_rankings