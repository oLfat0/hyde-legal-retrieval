"""
hyde_hybrid_v2.py
-----------------
Experimento v2 — HyDE + Hybrid Retrieval

Dense  : media de 5 embeddings de ementas hipotetrias  f(mean(h1..h5))
Sparse : BM25(descricao_original, corpus_ementas)  <- sempre query original
Fusao  : RRF

PRE-REQUISITOS:
  1. python -m src.hyde_v2
  2. python -m src.hyde_embedder_v2

Saida: results_v2/hyde_hybrid_v2.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets_v2        import QUERIES_V2_PATH, RESULTS_V2_DIR
from src.embedder_v2      import load_model, build_index, dense_search
from src.hyde_embedder_v2  import run_full_pipeline
from src.retriever_v2      import build_bm25_index, sparse_search, reciprocal_rank_fusion
from src.evaluator         import evaluate, print_results, save_results

CONFIG_NAME = "hyde_hybrid_v2"
TOP_K, KS   = 10, [1, 5, 10]

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : {CONFIG_NAME}")
    print(f"  Query       : media de 5 ementas hipotetrias (HyDE v2)")
    print(f"  BM25        : descricao original (sempre qi)")
    print(f"  Recuperacao : Hybrid (Dense + BM25 + RRF)")
    print(f"{'#'*55}\n")

    mean_embeddings, cdac_hyde_order = run_full_pipeline()

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_map    = {q["cdacordao"]: q["query"] for q in queries}
    assert set(cdac_hyde_order) == set(query_map.keys())

    query_texts  = [query_map[cid] for cid in cdac_hyde_order]  # descricao original para BM25
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
    save_results(results, CONFIG_NAME, output_dir=RESULTS_V2_DIR)
    return results

if __name__ == "__main__":
    run()