"""
hyde_dense_v2.py
----------------
Experimento v2 — HyDE + Dense Retrieval

Query      : media de 5 embeddings de ementas hipotetrias geradas da descricao
Corpus     : ementas reais (palavras-chave)
Recuperacao: FAISS IndexFlatIP

PRE-REQUISITOS:
  1. python -m src.hyde_v2
  2. python -m src.hyde_embedder_v2

Saida: results_v2/hyde_dense_v2.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets_v2       import QUERIES_V2_PATH, RESULTS_V2_DIR
from src.embedder_v2     import load_model, build_index, dense_search
from src.hyde_embedder_v2 import run_full_pipeline
from src.evaluator        import evaluate, print_results, save_results

CONFIG_NAME = "hyde_dense_v2"
TOP_K, KS   = 10, [1, 5, 10]

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : {CONFIG_NAME}")
    print(f"  Query       : media de 5 ementas hipotetrias (HyDE v2, T=0.7)")
    print(f"  Corpus      : ementas reais (palavras-chave)")
    print(f"  Recuperacao : Dense Retrieval (FAISS)")
    print(f"{'#'*55}\n")

    mean_embeddings, cdac_hyde_order = run_full_pipeline()
    print(f"[{CONFIG_NAME}] mean_embeddings shape: {mean_embeddings.shape}")

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    assert set(cdac_hyde_order) == {q["cdacordao"] for q in queries}

    relevant_ids = cdac_hyde_order

    model, tokenizer = load_model()
    index, cdac_list = build_index(model, tokenizer)

    print(f"[{CONFIG_NAME}] Dense retrieval com HyDE averaging (top-{TOP_K})...")
    rankings = dense_search(mean_embeddings, index, cdac_list, top_k=TOP_K)

    results = evaluate(rankings, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_V2_DIR)
    return results

if __name__ == "__main__":
    run()