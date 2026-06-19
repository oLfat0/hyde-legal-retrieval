"""
hyde_dense.py
-------------
Experimento 2: HyDE + Dense Retrieval (multi-round averaging)

Representacao da query:
    Para cada rodada n: hi_n = LLM_hyde(qi)
    vi_hyde = L2_normalize( mean(f(h^1_i), ..., f(h^10_i)) )

Recuperacao: busca densa (FAISS IndexFlatIP)

PRE-REQUISITOS (nesta ordem):
  1. python -m src.hyde              -> gera hyde_docs_{01..10}.json
  2. python -m src.hyde_embedder     -> gera hyde_embeds_{01..10}.npy + mean.npy

Saida: results/hyde_dense.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embedder, evaluator, hyde_embedder
from src.assets import QUERIES_PATH

CONFIG_NAME = "hyde_dense"
TOP_K       = 10
KS          = [1, 5, 10]


def run() -> dict:
    print(f"\n{'#'*55}")
    print(f"  Experimento: {CONFIG_NAME}")
    print(f"  Representacao : media de {hyde_embedder.hyde_module.N_ROUNDS} embeddings HyDE (T=0.7)")
    print(f"  Recuperacao   : Dense Retrieval (FAISS)")
    print(f"{'#'*55}\n")

    # 1. Carrega embeddings medios pre-computados
    print("[hyde_dense] Carregando embeddings medios HyDE...")
    mean_embeddings, cdacordao_hyde_order = hyde_embedder.run_full_pipeline()
    print(f"[hyde_dense] mean_embeddings shape: {mean_embeddings.shape}")

    # 2. Carrega queries para obter relevant_ids na mesma ordem
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Garante que a ordem dos cdacordao bate entre hyde_embedder e queries
    query_cdac_set = {q["cdacordao"] for q in queries}
    assert set(cdacordao_hyde_order) == query_cdac_set, (
        "ERRO: cdacordao nos embeddings HyDE divergem das queries!"
    )

    # relevant_ids na mesma ordem do hyde_embedder (posicao = linha no array)
    relevant_ids = cdacordao_hyde_order

    # 3. Carrega indice FAISS (reutiliza cache de baseline_dense)
    model, tokenizer = embedder.load_model()
    index, cdacordao_list = embedder.build_index(model, tokenizer)

    # 4. Dense retrieval com embedding medio
    print(f"[hyde_dense] Executando dense retrieval (top-{TOP_K})...")
    rankings = embedder.dense_search(
        mean_embeddings, index, cdacordao_list, top_k=TOP_K
    )

    # 5. Avaliacao
    print("[hyde_dense] Calculando metricas...")
    results = evaluator.evaluate(rankings, relevant_ids, ks=KS)

    evaluator.print_results(results, CONFIG_NAME)
    evaluator.save_results(results, CONFIG_NAME)
    return results


if __name__ == "__main__":
    run()