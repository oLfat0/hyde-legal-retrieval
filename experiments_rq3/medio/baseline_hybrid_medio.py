"""baseline_hybrid_medio.py — RQ3: Baseline + Hybrid, cenario medio"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.retriever_v2 import build_bm25_index, sparse_search, reciprocal_rank_fusion

from src.embedder_v2     import load_model, build_index, encode_texts, dense_search
from src.evaluator       import evaluate, print_results, save_results

QUERIES_PATH = Path("data/queries/queries_rq3_medio.json")
CONFIG_NAME  = "baseline_hybrid_rq3_medio"
TOP_K, KS    = 10, [1, 5, 10]
RESULTS_DIR  = "results_rq3"

def run() -> dict:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\n{'#'*55}")
    print(f"  RQ3 — Baseline + Hybrid | cenario: medio")
    print(f"  Indice: {len(queries)} documentos (RQ3)")
    print(f"{'#'*55}\n")

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]

    model, tokenizer = load_model()
    index, cdac_list = build_index(model, tokenizer)
    bm25, bm25_cdacs = build_bm25_index()
    assert cdac_list == bm25_cdacs, "Ordem FAISS vs BM25 diverge!"

    emb      = encode_texts(query_texts, model, tokenizer, desc=f"Enc baseline hybrid medio")
    dense_r  = dense_search(emb, index, cdac_list, top_k=TOP_K)
    sparse_r = sparse_search(query_texts, bm25, cdac_list, top_k=TOP_K)
    hybrid_r = reciprocal_rank_fusion(dense_r, sparse_r, top_k=TOP_K)

    results = evaluate(hybrid_r, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_DIR)
    return results

if __name__ == "__main__":
    run()