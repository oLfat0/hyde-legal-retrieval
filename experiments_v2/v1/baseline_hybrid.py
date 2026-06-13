"""
baseline_hybrid.py
------------------
Experimento 3: Baseline + Hybrid Retrieval

Representacao da query : embedding direto do mini-resumo
    vi_base = f(qi)

Recuperacao             : Hibrida (Dense + BM25 + RRF)
    R_dense   = sort(sim(vi_base, ej))          <- semantico
    R_lex     = BM25(qi, corpus)                <- lexical com query original
    R_hybrid  = RRF(R_dense, R_lex)             <- fusao

Saida: results/baseline_hybrid.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets import QUERIES_PATH
from src import embedder, evaluator, retriever

# -- Configuracoes do experimento ----------------------------------------------
CONFIG_NAME = "baseline_hybrid"
TOP_K       = 10
KS          = [1, 5, 10]


def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : embedding direto do mini-resumo")
    print(f"  Recuperacao   : Hybrid Retrieval (Dense + BM25 + RRF)")
    print(f"{'#'*55}\n")

    # 1. Carrega queries
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]
    print(f"[baseline_hybrid] {len(queries)} queries carregadas")

    # 2. Carrega modelo + indice FAISS (reutiliza cache)
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)

    # 3. Carrega indice BM25 (reutiliza cache ou constroi)
    bm25, bm25_cdacordao_list = retriever.build_bm25_index()

    # Sanidade: mesma ordem de documentos nos dois indices
    assert cdacordao_list == bm25_cdacordao_list, (
        "ERRO: ordem dos documentos no FAISS e no BM25 divergem!\n"
        "Reconstrua ambos os indices com force_rebuild=True."
    )

    # 4. Embeddings das queries (vi_base = f(qi))
    print("[baseline_hybrid] Computando embeddings das queries (baseline)...")
    query_embeddings = embedder.encode_texts(
        query_texts, model, tokenizer, desc="Encoding queries (baseline)"
    )

    # 5. Ranking denso
    print(f"[baseline_hybrid] Dense retrieval (top-{TOP_K})...")
    dense_rankings = embedder.dense_search(
        query_embeddings, index, cdacordao_list, top_k=TOP_K
    )

    # 6. Ranking lexical BM25 (SEMPRE com query original qi)
    print(f"[baseline_hybrid] BM25 sparse retrieval (top-{TOP_K})...")
    sparse_rankings = retriever.sparse_search(
        query_texts, bm25, cdacordao_list, top_k=TOP_K
    )

    # 7. Fusao RRF
    print("[baseline_hybrid] Aplicando RRF...")
    hybrid_rankings = retriever.reciprocal_rank_fusion(
        dense_rankings, sparse_rankings, top_k=TOP_K
    )

    # 8. Avaliacao
    print("[baseline_hybrid] Calculando metricas...")
    results = evaluator.evaluate(hybrid_rankings, relevant_ids, ks=KS)

    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)

    return results


if __name__ == "__main__":
    run()