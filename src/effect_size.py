"""
effect_size.py
--------------
Tamanho de efeito (r de Wilcoxon) e intervalos de confianca para a comparacao
pareada entre duas configuracoes experimentais.

Complementa o src/wilcoxon_test.py: enquanto aquele responde "a diferenca e
significativa?", este responde "a diferenca e GRANDE?". Com N = 501 consultas
ate ganhos pequenos ficam significativos, entao o p-valor sozinho nao comunica
a magnitude pratica do ganho.

Efeito reportado: r = |Z| / sqrt(N), convencao de Rosenthal (1991).

NOTA ESTATISTICA IMPORTANTE (empates)
-------------------------------------
A estatistica W do Wilcoxon e construida SOMENTE sobre os pares com diferenca
diferente de zero (o scipy, com zero_method="wilcox", descarta os empates antes
de ranquear). Logo:

  * mean_w e std_w OBRIGATORIAMENTE usam n_efetivo (pares sem empate).
    Plugar N = 501 ali produz um Z que nao corresponde a distribuicao nula de
    W (o script imprime esse valor apenas para deixar o erro visivel).

  * o denominador sqrt(N) do r e que admite as duas leituras, e as duas sao
    reportadas lado a lado:
       (a) N = total de pares            -> efeito diluido pelos empates
       (b) N = n_efetivo (sem empates)   -> efeito entre quem de fato mudou

    (a) e a leitura conservadora e a recomendada para o artigo.

Uso:
  # RQ1 — HyDE Dense vs Baseline Dense (default)
  python -m src.effect_size

  # RQ2 — HyDE Hybrid vs Baseline Hybrid
  python -m src.effect_size --a hyde_hybrid_v2 --b baseline_hybrid_v2

  # sem bootstrap (mais rapido)
  python -m src.effect_size --boot 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

DEFAULT_RESULTS_DIR = Path("results_v2")


def _load_per_query_rr(config_name: str, results_dir: Path) -> list[float]:
    """
    Carrega a lista de reciprocal ranks (campo 'rr') na ordem das queries.

    Estrutura esperada (do evaluator.py):
      {"per_query": [{"rank": N, "rr": X, "recall@1": ...}, ...]}
    """
    path = results_dir / f"{config_name}.json"
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


def _normal_params(n: int) -> tuple[float, float]:
    """Media e desvio da distribuicao nula de W para n pares ranqueados."""
    mean_w = n * (n + 1) / 4
    std_w = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
    return mean_w, std_w


def _tie_corrected_std(n: int, abs_diff: np.ndarray) -> float:
    """
    Desvio de W com correcao para empates nos POSTOS (valores de |d_i| iguais).

    E a versao que o scipy usa internamente:
      std = sqrt( (n(n+1)(2n+1) - sum(t^3 - t)/2) / 24 )
    onde t sao os tamanhos dos grupos de |d_i| empatados.
    """
    _, counts = np.unique(abs_diff, return_counts=True)
    tie_term = float(((counts ** 3) - counts).sum())
    return float(np.sqrt((n * (n + 1) * (2 * n + 1) - tie_term / 2) / 24))


def _abs_z(a: np.ndarray, b: np.ndarray) -> float | None:
    """|Z| (com correcao de empates) de uma amostra pareada — usado no bootstrap."""
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return None
    ranks = rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    mean_w, _ = _normal_params(n)
    std_w = _tie_corrected_std(n, np.abs(d))
    if std_w == 0:
        return None
    return abs((w_plus - mean_w) / std_w)


def classify(r: float) -> str:
    """Convencao de Cohen/Rosenthal para o r de Wilcoxon."""
    if r >= 0.50:
        return "large effect size"
    if r >= 0.30:
        return "medium effect size"
    return "small effect size"


def run(
    config_a: str,
    config_b: str,
    alpha: float = 0.05,
    n_boot: int = 10000,
    seed: int = 42,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    conf = int(round((1 - alpha) * 100))

    print(f"\n{'='*60}")
    print(f"  Tamanho de efeito de Wilcoxon (r = |Z| / sqrt(N))")
    print(f"  dir: {results_dir}")
    print(f"  A: {config_a}")
    print(f"  B: {config_b}")
    print(f"  alpha = {alpha}  (IC de {conf}%)")
    print(f"{'='*60}\n")

    a = np.array(_load_per_query_rr(config_a, results_dir))
    b = np.array(_load_per_query_rr(config_b, results_dir))

    if len(a) != len(b):
        raise ValueError(
            f"Numero de queries diverge: A={len(a)}, B={len(b)}. "
            "Os experimentos devem ter sido rodados sobre o mesmo conjunto de queries."
        )

    n_total = len(a)
    diff = a - b
    nz = diff[diff != 0]
    n_eff = len(nz)
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())

    if n_eff == 0:
        print("[erro] todas as diferencas sao zero: tamanho de efeito indefinido.")
        return

    # ---------------------------------------------------------------- amostra
    print("  --- Amostra ---------------------------------------------")
    print(f"  N consultas pareadas  : {n_total}")
    print(f"  MRR(A)                : {a.mean():.4f}")
    print(f"  MRR(B)                : {b.mean():.4f}")
    print(f"  Diferenca de MRR (A-B): {a.mean() - b.mean():+.4f}")
    print(f"  A > B em              : {n_pos} consultas")
    print(f"  A < B em              : {n_neg} consultas")
    print(f"  Empates               : {n_tie} consultas")
    print(f"  n efetivo (sem empate): {n_eff}")
    print()

    # ------------------------------------------------------------- teste base
    stat, p_two = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    print("  --- Wilcoxon signed-rank --------------------------------")
    print(f"  Estatistica W         : {stat:.1f}")
    print(f"  p-valor (bicaudal)    : {p_two:.3e}")
    print()

    # ------------------------------------------------------------ Z manual(is)
    # Usamos W+ (soma dos postos positivos). Como W+ + W- = n(n+1)/2 = 2*mean_w,
    # |W - mean_w| e identico para W+ e W-: o |Z| nao depende de qual dos dois
    # o scipy devolve como 'statistic'.
    ranks = rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    mean_w, std_w = _normal_params(n_eff)
    z_plain = (w_plus - mean_w) / std_w

    std_w_tc = _tie_corrected_std(n_eff, np.abs(nz))
    z_tc = (w_plus - mean_w) / std_w_tc

    print("  --- Aproximacao normal (n = n_efetivo) ------------------")
    print(f"  W+ (soma dos postos +): {w_plus:.1f}")
    print(f"  mean_w = n(n+1)/4     : {mean_w:.1f}")
    print(f"  std_w (formula pura)  : {std_w:.4f}")
    print(f"  std_w (c/ correcao de empates nos postos): {std_w_tc:.4f}")
    print(f"  Z manual (formula)    : {z_plain:+.4f}")
    print(f"  Z manual (corrigido)  : {z_tc:+.4f}")
    print()

    # ------------------------------------------------------- checagem cruzada
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                   method="approx")
    z_scipy = float(res.zstatistic)
    d_tc = abs(abs(z_scipy) - abs(z_tc))
    d_plain = abs(abs(z_scipy) - abs(z_plain))
    n_rep = n_eff - len(np.unique(np.abs(nz)))

    print("  --- Checagem cruzada com scipy (method='approx') --------")
    print(f"  Z do scipy            : {z_scipy:+.4f}")
    if d_tc < 1e-6:
        print(f"  OK: |Z| manual corrigido CONFERE com o scipy (dif = {d_tc:.2e})")
    else:
        print(f"  [ALERTA] |Z| manual corrigido DIVERGE do scipy: dif = {d_tc:.4f}")
        print("           investigue antes de reportar qualquer numero daqui.")
    if d_plain >= 1e-6:
        print(f"  [nota] a formula pura difere do scipy em {d_plain:.4f}: o scipy")
        print(f"         desconta os empates de |d_i| do std_w, e ha {n_rep} valores")
        print("         de |d_i| repetidos. A diferenca e esperada, nao e um bug.")
    print()

    # ---------------------------------------------------------- Z ingenuo (N)
    mean_bad, std_bad = _normal_params(n_total)
    z_bad = (w_plus - mean_bad) / std_bad
    print("  --- Por que mean_w/std_w NAO podem usar N = total -------")
    print(f"  Com n = {n_total} em mean_w/std_w: Z = {z_bad:+.4f}")
    print(f"  contra Z = {z_tc:+.4f} do scipy. W foi calculado sobre {n_eff} postos,")
    print(f"  entao a distribuicao nula de W e a de n = {n_eff}, nao a de {n_total}.")
    print("  Este Z NAO deve ser reportado; fica impresso apenas como aviso.")
    print()

    # --------------------------------------------------------- tamanho efeito
    abs_z = abs(z_tc)
    r_total = abs_z / np.sqrt(n_total)
    r_eff = abs_z / np.sqrt(n_eff)

    print("  --- Tamanho de efeito r = |Z| / sqrt(N) -----------------")
    print(f"  (a) N = {n_total:>3} (todos os pares)    : r = {r_total:.4f}  -> {classify(r_total)}")
    print(f"  (b) N = {n_eff:>3} (pares sem empate)  : r = {r_eff:.4f}  -> {classify(r_eff)}")
    print("  Convencao: r >= 0.50 large | 0.30 <= r < 0.50 medium | r < 0.30 small")
    print("  Recomendado para o artigo: (a), a leitura conservadora.")
    print()

    # ------------------------------------------------------------- bootstrap
    ci_delta = ci_r = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_total, size=(n_boot, n_total))
        boot_delta = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        boot_r = []
        for row in idx:
            z = _abs_z(a[row], b[row])
            if z is not None:
                boot_r.append(z / np.sqrt(n_total))
        lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
        ci_delta = (float(np.percentile(boot_delta, lo_q)),
                    float(np.percentile(boot_delta, hi_q)))
        ci_r = (float(np.percentile(boot_r, lo_q)),
                float(np.percentile(boot_r, hi_q)))

        print(f"  --- Intervalos de confianca ({conf}%, bootstrap {n_boot}x) ------")
        print(f"  Diferenca de MRR (A-B): {a.mean() - b.mean():+.4f}  "
              f"IC [{ci_delta[0]:+.4f}, {ci_delta[1]:+.4f}]")
        print(f"  r com N = {n_total}         : {r_total:.4f}   "
              f"IC [{ci_r[0]:.4f}, {ci_r[1]:.4f}]")
        print(f"  Faixa do IC de r      : {classify(ci_r[0])} .. {classify(ci_r[1])}")
        print(f"  (bootstrap percentil sobre as {n_total} consultas, seed={seed})")
        print()

    # ---------------------------------------------------------------- artigo
    if p_two < 0.001:
        p_report = "p < 0{,}001"
    else:
        p_report = f"p = {p_two:.3f}".replace(".", "{,}")
    r_latex = f"{r_total:.2f}".replace(".", "{,}")
    z_latex = f"{z_tc:.2f}".replace(".", "{,}")
    print("  --- Para o artigo ---------------------------------------")
    print(f"  $W = {stat:.0f}$, $Z = {z_latex}$, ${p_report}$, "
          f"$r = {r_latex}$ ({classify(r_total)})")
    if ci_delta is not None:
        d_lat = f"{a.mean() - b.mean():+.3f}".replace(".", "{,}")
        lo = f"{ci_delta[0]:+.3f}".replace(".", "{,}")
        hi = f"{ci_delta[1]:+.3f}".replace(".", "{,}")
        print(f"  $\\Delta$MRR $= {d_lat}$, IC{conf}\\% $[{lo};\\ {hi}]$")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tamanho de efeito r de Wilcoxon entre duas configuracoes."
    )
    parser.add_argument("--a", default="hyde_dense_v2", help="config A")
    parser.add_argument("--b", default="baseline_dense_v2", help="config B")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--boot", type=int, default=10000,
                        help="reamostragens do bootstrap (0 desliga)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dir", default=str(DEFAULT_RESULTS_DIR),
                        help="diretorio dos resultados (ex: results_rq3/longo)")
    args = parser.parse_args()
    run(args.a, args.b, alpha=args.alpha, n_boot=args.boot, seed=args.seed,
        results_dir=Path(args.dir))
