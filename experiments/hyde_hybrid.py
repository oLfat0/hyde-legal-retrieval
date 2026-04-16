"""
hyde_hybrid.py
--------------
Experimento 4: HyDE + Hybrid Retrieval

Representacao da query : embedding do documento hipotetico
    hi       = LLM_hyde(qi)
    vi_hyde  = f(hi)

Recuperacao             : Hibrida (Dense + BM25 + RRF)
    R_dense   = sort(sim(vi_hyde, ej))          <- semantico com HyDE
    R_lex     = BM25(qi, corpus)                <- lexical com query ORIGINAL
    R_hybrid  = RRF(R_dense, R_lex)             <- fusao

Nota arquitetural: o BM25 usa qi original (nao hi). Isso e intencional:
preserva a precisao lexical da query real enquanto o lado denso
se beneficia da expansao semantica do HyDE. Este design espelha
o diagrama do artigo (Query(q) -> BM25 | HyDE -> Embedding -> Dense).

PRE-REQUISITO: data/hyde_docs/hyde_docs.json (gere com python -m src.hyde)

Saida: results/hyde_hybrid.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedder, evaluator, hyde, retriever

# -- Configuracoes do experimento ----------------------------------------------
CONFIG_NAME = "hyde_hybrid"
TOP_K       = 10
KS          = [1, 5, 10]


def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : embedding do doc hipotetico (HyDE)")
    print(f"  Recuperacao   : Hybrid Retrieval (Dense + BM25 + RRF)")
    print(f"  BM25          : sempre usa query original qi")
    print(f"{'#'*55}\n")

    # 1. Carrega documentos HyDE pre-gerados
    print("[hyde_hybrid] Carregando documentos hipotetricos pre-gerados...")
    hyde_docs = hyde.load_hyde_docs()
    print(f"[hyde_hybrid] {len(hyde_docs)} documentos HyDE carregados")

    hyde_texts    = [doc["hyde_doc"] for doc in hyde_docs]
    query_texts   = [doc["query"]    for doc in hyde_docs]  # qi original para BM25
    relevant_ids  = [doc["cdacordao"] for doc in hyde_docs]

    # 2. Carrega modelo + indice FAISS
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)

    # 3. Carrega indice BM25
    bm25, bm25_cdacordao_list = retriever.build_bm25_index()

    assert cdacordao_list == bm25_cdacordao_list, (
        "ERRO: ordem dos documentos no FAISS e no BM25 divergem!\n"
        "Reconstrua ambos os indices com force_rebuild=True."
    )

    # 4. Embeddings dos documentos hipotetricos (vi_hyde = f(hi))
    print("[hyde_hybrid] Computando embeddings dos docs hipotetricos...")
    hyde_embeddings = embedder.encode_texts(
        hyde_texts, model, tokenizer, desc="Encoding HyDE docs"
    )

    # 5. Ranking denso com vi_hyde
    print(f"[hyde_hybrid] Dense retrieval com HyDE embeddings (top-{TOP_K})...")
    dense_rankings = embedder.dense_search(
        hyde_embeddings, index, cdacordao_list, top_k=TOP_K
    )

    # 6. Ranking lexical BM25 com qi ORIGINAL
    print(f"[hyde_hybrid] BM25 sparse retrieval com queries originais (top-{TOP_K})...")
    sparse_rankings = retriever.sparse_search(
        query_texts, bm25, cdacordao_list, top_k=TOP_K
    )

    # 7. Fusao RRF
    print("[hyde_hybrid] Aplicando RRF...")
    hybrid_rankings = retriever.reciprocal_rank_fusion(
        dense_rankings, sparse_rankings, top_k=TOP_K
    )

    # 8. Avaliacao
    print("[hyde_hybrid] Calculando metricas...")
    results = evaluator.evaluate(hybrid_rankings, relevant_ids, ks=KS)

    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)

    return results


if __name__ == "__main__":
    run()