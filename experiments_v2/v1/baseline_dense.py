"""
baseline_dense.py
-----------------
Experimento 1: Baseline + Dense Retrieval

Estrategia de representacao : embedding direto do mini-resumo
    vi_base = f(qi)

Estrategia de recuperacao   : busca densa (FAISS IndexFlatIP)
    R_dense = sort(sim(vi_base, ej))

Saida: results/baseline_dense.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets import QUERIES_PATH
from src import embedder, evaluator

# -- Configuracoes do experimento ----------------------------------------------
CONFIG_NAME = "baseline_dense"
TOP_K       = 10       # candidatos retornados por query
KS          = [1, 5, 10]


def run() -> dict:
    """Executa o experimento Baseline + Dense Retrieval."""

    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : embedding direto do mini-resumo")
    print(f"  Recuperacao   : Dense Retrieval (FAISS)")
    print(f"{'#'*55}\n")

    # 1. Carrega modelo e indice (indice é reutilizado entre experimentos)
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)

    # 2. Carrega queries
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    query_texts  = [q["query"]     for q in queries]
    relevant_ids = [q["cdacordao"] for q in queries]

    print(f"[baseline_dense] {len(queries)} queries carregadas")

    # 3. Computa embeddings das queries (vi_base = f(qi))
    print("[baseline_dense] Computando embeddings das queries (baseline)...")
    query_embeddings = embedder.encode_texts(
        query_texts,
        model,
        tokenizer,
        desc="Encoding queries (baseline)",
    )

    # 4. Busca densa
    print(f"[baseline_dense] Executando dense retrieval (top-{TOP_K})...")
    rankings = embedder.dense_search(
        query_embeddings,
        index,
        cdacordao_list,
        top_k=TOP_K,
    )

    # 5. Avaliacao
    print("[baseline_dense] Calculando metricas...")
    results = evaluator.evaluate(rankings, relevant_ids, ks=KS)

    # 6. Exibe e salva
    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)

    return results


if __name__ == "__main__":
    run()