"""
evaluator.py
------------
Implementacao das metricas de recuperacao para o protocolo de auto-retrieval.

Protocolo:
  - Cada query qi tem EXATAMENTE um documento relevante: o di com mesmo cdacordao
  - rel(qi, dj) = 1 se cdacordao_i == cdacordao_j, 0 caso contrario

Metricas implementadas:
  - Recall@k    : proporcao de queries onde doc relevante aparece nas top-k posicoes
  - MRR         : Mean Reciprocal Rank
  - nDCG@k      : Normalized Discounted Cumulative Gain (single-relevant-doc)

Todas as funcoes recebem rankings como list[list[tuple[cdacordao, score]]].
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path


# -- Funcao auxiliar: posicao do doc relevante no ranking ----------------------

def _find_rank(ranking: list[tuple[str, float]], relevant_id: str) -> int | None:
    """
    Retorna a posicao 1-based do documento relevante no ranking.
    Retorna None se o documento nao estiver no ranking.
    """
    for pos, (cdacordao, _) in enumerate(ranking, start=1):
        if cdacordao == relevant_id:
            return pos
    return None


# -- Metricas individuais ------------------------------------------------------

def recall_at_k(rank: int | None, k: int) -> float:
    """1.0 se o doc relevante esta nas top-k posicoes, 0.0 caso contrario."""
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """1/rank se encontrado, 0.0 caso contrario."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def ndcg_at_k(rank: int | None, k: int) -> float:
    """
    nDCG@k para um unico documento relevante (relevancia binaria = 1).

    DCG  = 1 / log2(rank + 1)  se rank <= k, senao 0
    IDCG = 1 / log2(1 + 1) = 1   (doc relevante sempre na posicao 1 no ideal)
    nDCG = DCG / IDCG = DCG
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


# -- Avaliacao agregada --------------------------------------------------------

def evaluate(
    rankings: list[list[tuple[str, float]]],
    relevant_ids: list[str],
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    Calcula todas as metricas para N queries.

    Args:
        rankings     : lista de N rankings (saida do dense_search ou hybrid_search)
        relevant_ids : lista de N cdacordao — o doc relevante de cada query
        ks           : valores de k para Recall@k e nDCG@k

    Retorna dict com:
        - recall@k, mrr, ndcg@k  (valores agregados, medias sobre N queries)
        - per_query               (lista com metricas individuais por query)
        - n_queries               (total de queries avaliadas)
        - n_found                 (queries onde o doc relevante apareceu no ranking)
    """
    assert len(rankings) == len(relevant_ids), (
        f"rankings ({len(rankings)}) e relevant_ids ({len(relevant_ids)}) devem ter o mesmo tamanho"
    )

    per_query: list[dict] = []
    rr_sum    = 0.0
    recall_sums = {k: 0.0 for k in ks}
    ndcg_sums   = {k: 0.0 for k in ks}
    n_found     = 0

    for ranking, rel_id in zip(rankings, relevant_ids):
        rank = _find_rank(ranking, rel_id)

        if rank is not None:
            n_found += 1

        rr  = reciprocal_rank(rank)
        rr_sum += rr

        query_metrics: dict = {"rank": rank, "rr": round(rr, 6)}

        for k in ks:
            r = recall_at_k(rank, k)
            n = ndcg_at_k(rank, k)
            recall_sums[k] += r
            ndcg_sums[k]   += n
            query_metrics[f"recall@{k}"]  = r
            query_metrics[f"ndcg@{k}"]    = round(n, 6)

        per_query.append(query_metrics)

    n = len(rankings)

    results: dict = {
        "n_queries" : n,
        "n_found"   : n_found,
        "mrr"       : round(rr_sum / n, 6),
    }
    for k in ks:
        results[f"recall@{k}"] = round(recall_sums[k] / n, 6)
        results[f"ndcg@{k}"]   = round(ndcg_sums[k]   / n, 6)

    results["per_query"] = per_query

    return results


# -- Exibicao dos resultados ---------------------------------------------------

def print_results(results: dict, config_name: str) -> None:
    """Imprime tabela de resultados no terminal."""
    print(f"\n{'='*55}")
    print(f"  Resultados: {config_name}")
    print(f"{'='*55}")
    print(f"  Queries avaliadas : {results['n_queries']}")
    print(f"  Docs encontrados  : {results['n_found']} ({results['n_found']/results['n_queries']*100:.1f}%)")
    print(f"  {'Metrica':<20} {'Valor':>10}")
    print(f"  {'-'*32}")

    metricas = (
        [f"recall@{k}" for k in [1, 5, 10]] +
        ["mrr"] +
        [f"ndcg@{k}" for k in [1, 5, 10]]
    )
    for m in metricas:
        if m in results:
            print(f"  {m:<20} {results[m]:>10.4f}")

    print(f"{'='*55}\n")


# -- Persistencia dos resultados -----------------------------------------------

def save_results(results: dict, config_name: str, output_dir: str = "results") -> Path:
    """
    Salva os resultados em results/{config_name}.json.

    O arquivo inclui metadados de execucao (timestamp, config_name) e
    as metricas agregadas + per_query para analise estatistica posterior.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{config_name}.json"

    payload = {
        "config"    : config_name,
        "timestamp" : datetime.now().isoformat(),
        "metrics"   : {k: v for k, v in results.items() if k != "per_query"},
        "per_query" : results["per_query"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[evaluator] Resultados salvos em {output_file}")
    return output_file