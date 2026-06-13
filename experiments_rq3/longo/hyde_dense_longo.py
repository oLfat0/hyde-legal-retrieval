"""hyde_dense_longo.py — RQ3: HyDE + Dense, cenario longo"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embedder_v2        import load_model, build_index, dense_search
from src.hyde_embedder_rq3  import run_pipeline
from src.evaluator          import evaluate, print_results, save_results

CONFIG_NAME = "hyde_dense_rq3_longo"
TOP_K, KS   = 10, [1, 5, 10]
RESULTS_DIR = "results_rq3"

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  RQ3 — HyDE + Dense | cenario: longo")
    print(f"  Indice: 501 documentos (RQ3)")
    print(f"{'#'*55}\n")

    mean_emb, cdac_order = run_pipeline(granul="longo")

    model, tokenizer = load_model()
    index, cdac_list = build_index(model, tokenizer)

    rankings = dense_search(mean_emb, index, cdac_list, top_k=TOP_K)
    results  = evaluate(rankings, cdac_order, ks=KS)

    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_DIR)
    return results

if __name__ == "__main__":
    run()