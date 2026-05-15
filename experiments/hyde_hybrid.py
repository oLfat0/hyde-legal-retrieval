"""
hyde_hybrid.py
--------------
Experimento 4: HyDE + Hybrid Retrieval (multi-round averaging)

Representacao da query:
    vi_hyde = L2_normalize( mean(f(h^1_i), ..., f(h^10_i)) )

Recuperacao:
    R_dense  = sort(sim(vi_hyde, ej))        <- semantico com HyDE averaging
    R_lex    = BM25(qi, corpus)              <- lexical com query ORIGINAL
    R_hybrid = RRF(R_dense, R_lex)

Nota: BM25 SEMPRE usa qi original, nunca os docs hipotetricos.

PRE-REQUISITOS (nesta ordem):
  1. python -m src.hyde              -> gera hyde_docs_{01..10}.json
  2. python -m src.hyde_embedder     -> gera hyde_embeds_{01..10}.npy + mean.npy

Saida: results/hyde_hybrid.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedder, evaluator, hyde_embedder, retriever
from src.assets import QUERIES_PATH

CONFIG_NAME = "hyde_hybrid"
TOP_K       = 10
KS          = [1, 5, 10]


def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : media de {hyde_embedder.hyde_module.N_ROUNDS} embeddings HyDE (T=0.7)")
    print(f"  Recuperacao   : Hybrid Retrieval (Dense + BM25 + RRF)")
    print(f"  BM25          : sempre usa query original qi")
    print(f"{'#'*55}\n")

    # 1. Carrega embeddings medios
    print("[hyde_hybrid] Carregando embeddings medios HyDE...")
    mean_embeddings, cdacordao_hyde_order = hyde_embedder.run_full_pipeline()

    # 2. Carrega queries para BM25 (precisa do texto original qi)
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Mapeia cdacordao -> query text para reordenar igual ao hyde_embedder
    query_map = {q["cdacordao"]: q["query"] for q in queries}

    assert set(cdacordao_hyde_order) == set(query_map.keys()), (
        "ERRO: cdacordao nos embeddings HyDE divergem das queries!"
    )

    # query_texts e relevant_ids na mesma ordem do hyde_embedder
    query_texts  = [query_map[cid]     for cid in cdacordao_hyde_order]
    relevant_ids = cdacordao_hyde_order

    # 3. Carrega indices
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)
    bm25, bm25_cdacordao_list = retriever.build_bm25_index()

    assert cdacordao_list == bm25_cdacordao_list, (
        "ERRO: ordem dos documentos no FAISS e no BM25 divergem!\n"
        "Reconstrua ambos com force_rebuild=True."
    )

    # 4. Dense retrieval com embedding medio HyDE
    print(f"[hyde_hybrid] Dense retrieval com HyDE averaging (top-{TOP_K})...")
    dense_rankings = embedder.dense_search(
        mean_embeddings, index, cdacordao_list, top_k=TOP_K
    )

    # 5. BM25 com queries originais
    print(f"[hyde_hybrid] BM25 sparse retrieval (top-{TOP_K})...")
    sparse_rankings = retriever.sparse_search(
        query_texts, bm25, cdacordao_list, top_k=TOP_K
    )

    # 6. RRF
    print("[hyde_hybrid] Aplicando RRF...")
    hybrid_rankings = retriever.reciprocal_rank_fusion(
        dense_rankings, sparse_rankings, top_k=TOP_K
    )

    # 7. Avaliacao
    print("[hyde_hybrid] Calculando metricas...")
    results = evaluator.evaluate(hybrid_rankings, relevant_ids, ks=KS)

    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)
    return results


if __name__ == "__main__":
    run()