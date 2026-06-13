"""
baseline_dense_v2.py
--------------------
Experimento v2 — Baseline + Dense Retrieval

Query      : descricao completa do processo  (vi_base = f(descricao))
Corpus     : ementas (palavras-chave)
Recuperacao: FAISS IndexFlatIP
Saida      : results_v2/baseline_dense_v2.json
"""

from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets_v2  import QUERIES_V2_PATH, RESULTS_V2_DIR
from src.embedder_v2 import load_model, build_index, encode_texts, dense_search
from src.evaluator   import evaluate, print_results, save_results

CONFIG_NAME = "baseline_dense_v2"
TOP_K, KS   = 10, [1, 5, 10]

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : {CONFIG_NAME}")
    print(f"  Query       : descricao do processo")
    print(f"  Corpus      : ementas (palavras-chave)")
    print(f"  Recuperacao : Dense Retrieval (FAISS)")
    print(f"{'#'*55}\n")

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_texts  = [q["query"]     for q in queries]   # descricao
    relevant_ids = [q["cdacordao"] for q in queries]

    print(f"[{CONFIG_NAME}] {len(queries)} queries (descricoes) carregadas")

    model, tokenizer  = load_model()
    index, cdac_list  = build_index(model, tokenizer)

    print(f"[{CONFIG_NAME}] Computando embeddings das descricoes...")
    query_embeddings = encode_texts(query_texts, model, tokenizer, desc="Enc descricoes (baseline)")

    print(f"[{CONFIG_NAME}] Dense retrieval (top-{TOP_K})...")
    rankings = dense_search(query_embeddings, index, cdac_list, top_k=TOP_K)

    results = evaluate(rankings, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_V2_DIR)
    return results

if __name__ == "__main__":
    run()