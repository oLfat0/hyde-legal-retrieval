"""
wilcoxon_test.py
----------------
Teste de Wilcoxon signed-rank pareado entre duas configuracoes experimentais.

Responde: a diferenca de desempenho entre (ex.) HyDE e Baseline e
estatisticamente significativa, ou pode ser fruto do acaso?

Compara as distribuicoes de reciprocal rank por query das duas configuracoes,
lidas do campo "per_query" dos arquivos results_v2/*.json.

O pareamento e POSICIONAL: ambos os experimentos processam as queries na mesma
ordem (a do queries_v2.json), entao a i-esima entrada de per_query corresponde
a mesma consulta nos dois arquivos.

Uso:
  # RQ1 — HyDE Dense vs Baseline Dense
  python -m src.wilcoxon_test --a hyde_dense_v2 --b baseline_dense_v2

  # RQ2 — HyDE Hybrid vs Baseline Hybrid
  python -m src.wilcoxon_test --a hyde_hybrid_v2 --b baseline_hybrid_v2

  # RQ1 — HyDE Dense vs BM25 puro
  python -m src.wilcoxon_test --a hyde_dense_v2 --b bm25_puro_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results_v2")


def _load_per_query_rr(config_name: str) -> list[float]:
    """
    Carrega a lista de reciprocal ranks (campo 'rr') na ordem das queries.

    Estrutura esperada (do evaluator.py):
      {"per_query": [{"rank": N, "rr": X, "recall@1": ...}, ...]}
    """
    path = RESULTS_DIR / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Resultado nao encontrado: {path}\n"
            f"Rode o experimento '{config_name}' antes."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_query" not in data:
        raise KeyError(f"'{config_name}' nao tem campo 'per_query'.")

    rr_list = []
    for pq in data["per_query"]:
        if "rr" in pq:
            rr_list.append(float(pq["rr"]))
        elif "rank" in pq:
            rank = pq["rank"]
            rr_list.append(1.0 / rank if rank and rank > 0 else 0.0)
        else:
            raise KeyError(f"per_query de '{config_name}' sem 'rr' nem 'rank'")
    return rr_list


def run_test(config_a: str, config_b: str, alpha: float = 0.05) -> None:
    print(f"\n{'='*60}")
    print(f"  Teste de Wilcoxon signed-rank (pareado)")
    print(f"  A: {config_a}")
    print(f"  B: {config_b}")
    print(f"  alpha = {alpha}")
    print(f"{'='*60}\n")

    a = np.array(_load_per_query_rr(config_a))
    b = np.array(_load_per_query_rr(config_b))

    if len(a) != len(b):
        raise ValueError(
            f"Numero de queries diverge: A={len(a)}, B={len(b)}. "
            "Os experimentos devem ter sido rodados sobre o mesmo conjunto de queries."
        )

    diff = a - b
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())

    print(f"  N consultas pareadas : {len(a)}")
    print(f"  MRR(A)               : {a.mean():.4f}")
    print(f"  MRR(B)               : {b.mean():.4f}")
    print(f"  Diferenca media      : {diff.mean():+.4f}")
    print(f"  A > B em             : {n_pos} consultas")
    print(f"  A < B em             : {n_neg} consultas")
    print(f"  Empates              : {n_tie} consultas")
    print()

    try:
        stat, p_two = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        print(f"[erro] {e}")
        print("  (ocorre se todas as diferencas forem zero)")
        return

    print(f"  Estatistica W        : {stat:.1f}")
    print(f"  p-valor (bicaudal)   : {p_two:.3e}")
    print()

    if p_two < alpha:
        vencedor = config_a if a.mean() > b.mean() else config_b
        print(f"  >> RESULTADO: diferenca SIGNIFICATIVA (p < {alpha})")
        print(f"     '{vencedor}' supera o outro de forma estatisticamente robusta.")
    else:
        print(f"  >> RESULTADO: diferenca NAO significativa (p >= {alpha})")

    # Formatacao LaTeX para o artigo
    if p_two < 0.001:
        p_report = "p < 0{,}001"
    else:
        p_report = f"p = {p_two:.3f}".replace(".", "{,}")
    print(f"\n  Para o artigo: $W = {stat:.0f}$, ${p_report}$")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import time
    start = time.time()
    if start:
        print("Timer iniciado...")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="config A (ex: hyde_dense_v2)")
    parser.add_argument("--b", required=True, help="config B (ex: baseline_dense_v2)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_test(args.a, args.b, alpha=args.alpha)

    end = time.time()
    print(f"Tempo: {(end-start)/60:.2f}min ({(end-start):.2f}s)")