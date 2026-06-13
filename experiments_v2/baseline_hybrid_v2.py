"""
baseline_hybrid_v2.py
---------------------
Experimento v2 — Baseline + Hybrid Retrieval

Dense  : f(descricao)
Sparse : BM25(descricao, corpus_ementas)  <- query original vs ementas
Fusao  : RRF
Saida  : results_v2/baseline_hybrid_v2.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets_v2   import QUERIES_V2_PATH, RESULTS_V2_DIR
from src.embedder_v2 import load_model, build_index, encode_texts, dense_search
from src.retriever_v2 import build_bm25_index, sparse_search, reciprocal_rank_fusion
from src.evaluator    import evaluate, print_results, save_results

CONFIG_NAME = "baseline_hybrid_v2"
TOP_K, KS   = 10, [1, 5, 10]

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : {CONFIG_NAME}")
    print(f"  Query       : descricao do processo")
    print(f"  Recuperacao : Hybrid (Dense + BM25 + RRF)")
    print(f"{'#'*55}\n")

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]

    model, tokenizer  = load_model()
    index, cdac_list  = build_index(model, tokenizer)
    bm25, bm25_cdacs  = build_bm25_index()

    assert cdac_list == bm25_cdacs, "Ordem dos docs diverge entre FAISS e BM25!"

    query_embeddings = encode_texts(query_texts, model, tokenizer, desc="Enc descricoes (baseline hybrid)")
    dense_rankings   = dense_search(query_embeddings, index, cdac_list, top_k=TOP_K)
    sparse_rankings  = sparse_search(query_texts, bm25, cdac_list, top_k=TOP_K)
    hybrid_rankings  = reciprocal_rank_fusion(dense_rankings, sparse_rankings, top_k=TOP_K)

    results = evaluate(hybrid_rankings, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_V2_DIR)
    return results

if __name__ == "__main__":
    run()