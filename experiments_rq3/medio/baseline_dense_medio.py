"""baseline_dense_medio.py — RQ3: Baseline + Dense, cenario medio"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embedder_v2     import load_model, build_index, encode_texts, dense_search
from src.evaluator       import evaluate, print_results, save_results

QUERIES_PATH = Path("data/queries/queries_rq3_medio.json")
CONFIG_NAME  = "baseline_dense_rq3_medio"
TOP_K, KS    = 10, [1, 5, 10]
RESULTS_DIR  = "results_rq3"

def run() -> dict:
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\n{'#'*55}")
    print(f"  RQ3 — Baseline + Dense | cenario: medio")
    print(f"  Indice: {len(queries)} documentos (RQ3)")
    print(f"{'#'*55}\n")
    

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]
    print(f"  {len(queries)} queries | extensao media: {sum(len(q.split()) for q in query_texts)//len(query_texts)} palavras")

    model, tokenizer  = load_model()
    index, cdac_list  = build_index(model, tokenizer)

    emb      = encode_texts(query_texts, model, tokenizer, desc=f"Enc baseline medio")
    rankings = dense_search(emb, index, cdac_list, top_k=TOP_K)
    results  = evaluate(rankings, relevant_ids, ks=KS)

    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_DIR)
    return results

if __name__ == "__main__":
    run()