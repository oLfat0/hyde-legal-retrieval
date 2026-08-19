"""
effect_size_batch.py
--------------------
Roda de uma vez as comparacoes pedidas pelo revisor (RQ2 e RQ3) e imprime uma
tabela compacta com p-valor e tamanho de efeito r.

Reaproveita as funcoes de src/effect_size.py. Para o detalhamento completo de
uma comparacao especifica, use o src/effect_size.py direto.

COLUNAS r_501 e r_efet
----------------------
  r_501  : r = |Z| / sqrt(501), com Z calculado sobre os pares sem empate.
           E a leitura conservadora — a recomendada para o artigo.
  r_efet : r = |Z| / sqrt(n_efetivo), so entre as consultas que mudaram.

Em ambas o Z vem da distribuicao nula de n_efetivo, que e a correta: o W do
Wilcoxon e construido apenas sobre os pares com diferenca != 0.

Uso:
  python -m src.effect_size_batch
  python -m src.effect_size_batch --granul-config hyde_hybrid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from src.effect_size import (
    _load_per_query_rr,
    _normal_params,
    _tie_corrected_std,
    classify,
)


def compute(a: np.ndarray, b: np.ndarray) -> dict:
    """Estatisticas do Wilcoxon pareado de A contra B."""
    n_total = len(a)
    diff = a - b
    nz = diff[diff != 0]
    n_eff = len(nz)

    out = {
        "n_total": n_total,
        "n_eff": n_eff,
        "n_tie": int((diff == 0).sum()),
        "n_pos": int((diff > 0).sum()),
        "n_neg": int((diff < 0).sum()),
        "mrr_a": float(a.mean()),
        "mrr_b": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
    }

    if n_eff == 0:
        out.update(p=1.0, w=0.0, z=0.0, r_total=0.0, r_eff=0.0, r_naive=0.0)
        return out

    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")

    ranks = rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    mean_w, _ = _normal_params(n_eff)
    z = (w_plus - mean_w) / _tie_corrected_std(n_eff, np.abs(nz))

    # replica exata da formula do orientador: N = total em mean_w/std_w, e a
    # estatistica que o scipy devolve (min(W+, W-)) no lugar de W+.
    mean_bad, std_bad = _normal_params(n_total)
    z_naive = (float(stat) - mean_bad) / std_bad

    out.update(
        p=float(p),
        w=float(stat),
        z=float(z),
        r_total=abs(z) / np.sqrt(n_total),
        r_eff=abs(z) / np.sqrt(n_eff),
        r_naive=abs(z_naive) / np.sqrt(n_total),
    )
    return out


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"  {p:.3f}"


def main(granul_config: str = "hyde_dense", alpha: float = 0.05) -> None:
    v2 = Path("results_v2")
    rq3 = Path("results_rq3")

    comparacoes = [
        # (rotulo, dir_A, config_A, dir_B, config_B)
        ("RQ2  Hybrid(HyDE+BM25) vs HyDE isolado",
         v2, "hyde_hybrid_v2", v2, "hyde_dense_v2"),
        ("RQ2  Hybrid(HyDE+BM25) vs BM25 puro",
         v2, "hyde_hybrid_v2", v2, "bm25_puro_v2"),
        (f"RQ3  Medium vs Short  [{granul_config}]",
         rq3 / "medio", f"{granul_config}_rq3_medio",
         rq3 / "curto", f"{granul_config}_rq3_curto"),
        (f"RQ3  Long vs Medium   [{granul_config}]",
         rq3 / "longo", f"{granul_config}_rq3_longo",
         rq3 / "medio", f"{granul_config}_rq3_medio"),
    ]

    print(f"\n{'='*94}")
    print("  Wilcoxon pareado + tamanho de efeito — comparacoes pedidas pelo revisor")
    print(f"  alpha = {alpha}   |   A vs B, sempre com A = configuracao mais rica")
    print(f"{'='*94}\n")

    cab = (f"  {'Comparacao':<40} {'MRR(A)':>7} {'MRR(B)':>7} {'delta':>8} "
           f"{'p':>8} {'r_501':>7} {'r_efet':>7}  {'veredito':<22}")
    print(cab)
    print(f"  {'-'*92}")

    resultados = []
    for rotulo, da, ca, db, cb in comparacoes:
        a = np.array(_load_per_query_rr(ca, da))
        b = np.array(_load_per_query_rr(cb, db))
        if len(a) != len(b):
            raise ValueError(f"{rotulo}: N difere ({len(a)} vs {len(b)})")
        st = compute(a, b)
        resultados.append((rotulo, st))

        sig = "SIGNIFICATIVO" if st["p"] < alpha else "NAO significativo"
        veredito = f"{sig}, {classify(st['r_total']).replace(' effect size','')}"
        print(f"  {rotulo:<40} {st['mrr_a']:>7.4f} {st['mrr_b']:>7.4f} "
              f"{st['delta']:>+8.4f} {_fmt_p(st['p']):>8} "
              f"{st['r_total']:>7.4f} {st['r_eff']:>7.4f}  {veredito:<22}")

    print()
    print("  --- Detalhe por comparacao ------------------------------------------------")
    for rotulo, st in resultados:
        print(f"\n  {rotulo}")
        print(f"    W = {st['w']:.1f} | Z = {st['z']:+.4f} | p = {st['p']:.3e}")
        print(f"    A > B: {st['n_pos']} | A < B: {st['n_neg']} | "
              f"empates: {st['n_tie']} | n_efetivo: {st['n_eff']}")
        print(f"    r (N=501) = {st['r_total']:.4f} -> {classify(st['r_total'])}")
        print(f"    r (N={st['n_eff']}) = {st['r_eff']:.4f} -> {classify(st['r_eff'])}")
        dif = abs(st["r_naive"] - st["r_total"])
        alerta = "  <-- MUDA A CATEGORIA" if classify(st["r_naive"]) != classify(st["r_total"]) else ""
        print(f"    [formula com N=501 em mean_w/std_w: r = {st['r_naive']:.4f} "
              f"({classify(st['r_naive'])}), erro de {dif:.4f}]{alerta}")

    print(f"\n{'='*94}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--granul-config", default="hyde_dense",
                        choices=["hyde_dense", "hyde_hybrid",
                                 "baseline_dense", "baseline_hybrid"],
                        help="configuracao usada na comparacao entre granularidades")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    main(granul_config=args.granul_config, alpha=args.alpha)
