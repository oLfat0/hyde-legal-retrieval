"""
hyde_dense.py
-------------
Experimento 2: HyDE + Dense Retrieval

Estrategia de representacao : embedding do documento hipotetico
    hi     = LLM_hyde(qi)
    vi_hyde = f(hi)

Estrategia de recuperacao   : busca densa (FAISS IndexFlatIP)
    R_dense = sort(sim(vi_hyde, ej))

PRE-REQUISITO: rodar src/hyde.py para gerar data/hyde_docs/hyde_docs.json

Saida: results/hyde_dense.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedder, evaluator, hyde

# -- Configuracoes do experimento ----------------------------------------------
CONFIG_NAME = "hyde_dense"
TOP_K       = 10
KS          = [1, 5, 10]


def run() -> dict:
    """Executa o experimento HyDE + Dense Retrieval."""

    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : embedding do documento hipotetico (HyDE)")
    print(f"  Recuperacao   : Dense Retrieval (FAISS)")
    print(f"{'#'*55}\n")

    # 1. Carrega documentos HyDE pre-gerados
    print("[hyde_dense] Carregando documentos hipotéticos pre-gerados...")
    hyde_docs = hyde.load_hyde_docs()
    print(f"[hyde_dense] {len(hyde_docs)} documentos HyDE carregados")

    hyde_texts   = [doc["hyde_doc"]  for doc in hyde_docs]
    relevant_ids = [doc["cdacordao"] for doc in hyde_docs]

    # 2. Carrega modelo e indice
    #    O mesmo indice do baseline_dense e reutilizado (build_index retorna do cache)
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)

    # 3. Computa embeddings dos documentos hipotetricos (vi_hyde = f(hi))
    print("[hyde_dense] Computando embeddings dos documentos hipotéticos...")
    hyde_embeddings = embedder.encode_texts(
        hyde_texts,
        model,
        tokenizer,
        desc="Encoding HyDE docs",
    )

    # 4. Busca densa com vi_hyde
    print(f"[hyde_dense] Executando dense retrieval (top-{TOP_K})...")
    rankings = embedder.dense_search(
        hyde_embeddings,
        index,
        cdacordao_list,
        top_k=TOP_K,
    )

    # 5. Avaliacao
    print("[hyde_dense] Calculando metricas...")
    results = evaluator.evaluate(rankings, relevant_ids, ks=KS)

    # 6. Exibe e salva
    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)

    return results


if __name__ == "__main__":
    run()