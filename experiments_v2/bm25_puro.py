"""
bm25_puro_v2.py — RQ1: BM25 lexical puro (sem componente denso)
Avalia a recuperacao usando APENAS o BM25 sobre o campo ementa,
com a descricao (corpo estruturado) como query. N=501.

Preenche a primeira linha da Tabela 1 (RQ1) do artigo.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever_v2 import build_bm25_index, sparse_search
from src.evaluator    import evaluate, print_results, save_results

QUERIES_PATH = Path("data/queries/queries_v2.json")
CONFIG_NAME  = "bm25_puro_v2"
TOP_K, KS    = 10, [1, 5, 10]
RESULTS_DIR  = "results_v2"

def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento : bm25_puro_v2")
    print(f"  Query       : descricao do processo")
    print(f"  Corpus      : ementas (palavras-chave)")
    print(f"  Recuperacao : BM25 Lexical Puro (sem dense)")
    print(f"{'#'*55}\n")

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]
    print(f"[bm25_puro_v2] {len(queries)} queries carregadas")

    bm25, bm25_cdacs = build_bm25_index()
    print(f"[bm25_puro_v2] BM25 retrieval (top-{TOP_K})...")
    rankings = sparse_search(query_texts, bm25, bm25_cdacs, top_k=TOP_K)

    results = evaluate(rankings, relevant_ids, ks=KS)
    print_results(results, CONFIG_NAME)
    save_results(results, CONFIG_NAME, output_dir=RESULTS_DIR)
    return results

if __name__ == "__main__":
    import time
    start = time.time()
    if start:
        print("Timer iniciado...")
    run()

    end = time.time()
    print(f"Tempo: {(end-start)/60:.2f}min ({(end-start):.2f}s)")
