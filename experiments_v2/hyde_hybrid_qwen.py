"""
hyde_hybrid_qwen.py
-------------------
Experimento Qwen — HyDE + Hybrid Retrieval (25 queries)

Dense  : media de 5 embeddings de ementas hipoteticas (Qwen2.5-14B)
Sparse : BM25(descricao_original, corpus_ementas)  <- sempre query original
Fusao  : RRF

PRE-REQUISITOS:
  1. python -m src.hyde_v2
  2. python -m src.hyde_embedder_v2  (run_full_pipeline_qwen)

Saida: results_qwen/hyde_hybrid_qwen.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets_v2        import QUERIES_V2_PATH, RESULTS_QWEN_DIR
from src.embedder_v2      import load_model, build_index, dense_search
from src.hyde_embedder_v2 import run_full_pipeline_qwen
from src.retriever_v2     import build_bm25_index, sparse_search, reciprocal_rank_fusion
from src.evaluator        import evaluate, print_results, save_results

CONFIG_NAME = "hyde_hybrid_qwen"
TOP_K, KS   = 10, [1, 5, 10]

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : {CONFIG_NAME}")
    print(f"  Query       : media de 5 ementas hipoteticas (Qwen2.5-14B)")
    print(f"  BM25        : descricao original (sempre qi)")
    print(f"  Recuperacao : Hybrid (Dense + BM25 + RRF)")
    print(f"{'#'*55}\n")

    mean_embeddings, cdac_hyde_order = run_full_pipeline_qwen()

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_map = {q["cdacordao"]: q["query"] for q in queries}
    assert set(cdac_hyde_order).issubset(set(query_map.keys()))

    query_texts  = [query_map[cid] for cid in cdac_hyde_order]
    relevant_ids = cdac_hyde_order

    model, tokenizer = load_model()
    index, cdac_list = build_index(model, tokenizer)
    bm25, bm25_cdacs = build_bm25_index()
    assert cdac_list == bm25_cdacs, "Ordem dos docs diverge entre FAISS e BM25!"

    dense_rankings  = dense_search(mean_embeddings, index, cdac_list, top_k=TOP_K)
    sparse_rankings = sparse_search(query_texts, bm25, cdac_list, top_k=TOP_K)
    hybrid_rankings = reciprocal_rank_fusion(dense_rankings, sparse_rankings, top_k=TOP_K)

    results = evaluate(hybrid_rankings, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_QWEN_DIR)
    return results

if __name__ == "__main__":
    run()
