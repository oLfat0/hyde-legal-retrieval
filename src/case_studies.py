"""
case_studies.py
---------------
Analise qualitativa: localiza e documenta os tres cenarios pedidos pelo revisor.

  CASO A — HyDE corrige falha do denso direto e do BM25 (vocabulary mismatch)
  CASO B — a fusao hibrida (HyDE + BM25) resgata o ranking do HyDE isolado
  CASO C — alucinacao do HyDE derruba o acordao-alvo (impacto negativo)

Para cada caso reporta o ranking do acordao-alvo em todos os metodos e os
TERMOS RESPONSAVEIS: os dispositivos legais e o jargao juridico que o HyDE
injetou (ou inventou), cruzados contra a ementa-gabarito.

ATENCAO — a recuperacao foi feita ate o top-10. Portanto "rank" = null
significa "fora do top-10", nao "rank alto". Filtros do tipo
`df['rank_dense'] > 10` retornam VAZIO porque comparacao com NaN e sempre
False; e preciso usar .isna() ou preencher com um sentinela.

Uso:
  python -m src.case_studies                 # relatorio dos 3 casos
  python -m src.case_studies --tabela        # so a tabela de rankings
  python -m src.case_studies --csv ranks.csv # exporta a tabela completa
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

FORA_DO_TOPK = 999  # sentinela para "nao recuperado no top-10"

METODOS = {
    "rank_bm25": "results_v2/bm25_puro_v2.json",
    "rank_dense": "results_v2/baseline_dense_v2.json",
    "rank_hyde": "results_v2/hyde_dense_v2.json",
    "rank_hybrid": "results_v2/hyde_hybrid_v2.json",
    "rank_basehyb": "results_v2/baseline_hybrid_v2.json",
}

QUERIES = "data/queries/queries_v2.json"
CORPUS = "data/corpus/ementas_v2.json"
HYDE_DOCS = "data/hyde_v2_docs/hyde_v2_docs_0{i}.json"


# --------------------------------------------------------------------- dados
def _ranks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [x["rank"] for x in json.load(f)["per_query"]]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_table() -> pd.DataFrame:
    """Tabela 501 x metodos com o rank do acordao-alvo em cada configuracao."""
    queries = _load_json(QUERIES)
    df = pd.DataFrame({
        "idx": range(len(queries)),
        "cdacordao": [q["cdacordao"] for q in queries],
        "numero_processo": [q["numero_processo"] for q in queries],
        "classe": [q["classe"] for q in queries],
    })
    for col, path in METODOS.items():
        df[col] = _ranks(path)
    # versao preenchida, para filtrar sem cair na armadilha do NaN
    for col in METODOS:
        df[col + "_f"] = df[col].fillna(FORA_DO_TOPK).astype(int)
    return df


def load_hyde_docs() -> dict[str, dict]:
    """Une os 5 shards de documentos hipoteticos, indexados por cdacordao."""
    docs = {}
    for i in range(1, 6):
        for d in _load_json(HYDE_DOCS.format(i=i)):
            docs[d["cdacordao"]] = d
    return docs


# ------------------------------------------------------------------- termos
CITACAO = re.compile(
    r"(?:art(?:igo)?s?\.?\s*\d+[\w\-\.\ºo°/]*"
    r"|tema\s+n?º?\s*[\d\.]+"
    r"|s[uú]mula\s+n?º?\s*[\d\.]+"
    r"|lei\s+n?º?\s*[\d\.]+(?:/\d+)?"
    r"|resolu[cç][aã]o\s+(?:cnj\s+)?n?º?\s*[\d\.\-/]+"
    r"|cpc|cdc|clt|ctn|cf/?88|c[oó]digo\s+civil|c[oó]digo\s+penal)",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def _norm_citacao(s: str) -> str:
    """
    Normaliza uma citacao para comparacao entre textos.

    Sem isto, "RESOLUCAO CNJ No 547/2024" (ementa) e "RESolucao CNJ 547/2024."
    (HyDE) nao casam, e o dispositivo seria falsamente acusado de alucinacao.
    Remove o marcador de numero, pontuacao de borda e ordinais.
    """
    s = _norm(s)
    s = re.sub(r"\bn[o°º]?\.?\s*", "", s)              # "nº", "n.", "no"
    s = re.sub(r"^(?:artigos?|arts?)\.?\s*", "art ", s)  # arts. 99 == art. 99
    s = re.sub(r"(\d)[o°º]\b", r"\1", s)                # "5º" -> "5"
    s = re.sub(r"[\s\.,;:]+$", "", s)                   # pontuacao final
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def citacoes(texto: str) -> set[str]:
    return {_norm_citacao(m.group(0)) for m in CITACAO.finditer(texto or "")}


def jaccard(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if (x or y) else 0.0


STOP = set(_norm(
    "de da do das dos a o as os e ou que em no na nos nas por para com sem "
    "ao aos as um uma uns umas se ser sao foi era pelo pela pelos pelas "
    "sobre entre ate apos antes qual quais cujo cuja este esta esse essa "
    "aquele aquela isso isto seu sua seus suas nao nem mais menos ja tambem "
    "i ii iii iv v vi caso em exame tese julgamento dispositivo"
).split())


def termos(texto: str, minlen: int = 5) -> set[str]:
    return {w for w in _norm(texto or "").split()
            if len(w) >= minlen and w not in STOP and not w.isdigit()}


# ------------------------------------------------------------------ relatorio
def _rk(v) -> str:
    return "fora do top-10" if pd.isna(v) else f"{int(v)}o"


def _linha_ranking(row) -> str:
    return (f"      BM25 puro      : {_rk(row['rank_bm25'])}\n"
            f"      Denso direto   : {_rk(row['rank_dense'])}\n"
            f"      HyDE denso     : {_rk(row['rank_hyde'])}\n"
            f"      Hibrido HyDE   : {_rk(row['rank_hybrid'])}\n"
            f"      Hibrido base   : {_rk(row['rank_basehyb'])}")


def _amostra(txt: str, n: int = 400) -> str:
    txt = re.sub(r"\s+", " ", txt or "").strip()
    return txt[:n] + ("..." if len(txt) > n else "")


def relatorio(df: pd.DataFrame, corpus_by_cd: dict, hyde_docs: dict,
              n_por_caso: int = 1) -> None:
    casos = [
        ("CASO A — HyDE supera o vocabulary mismatch",
         "denso direto e BM25 falham (fora do top-10); HyDE leva ao 1o lugar",
         (df.rank_dense_f > 10) & (df.rank_bm25_f > 10) & (df.rank_hyde_f == 1)),
        ("CASO B — Sinergia da hibridizacao (RRF)",
         "HyDE isolado fica abaixo do topo; a fusao com BM25 promove ao 1o lugar",
         (df.rank_hyde_f > 2) & (df.rank_hybrid_f == 1) & (df.rank_bm25_f <= 2)),
        ("CASO C — Alucinacao degrada a recuperacao",
         "HyDE cita dispositivo ausente da ementa-alvo E da consulta, e o "
         "ranking piora vs. denso direto",
         (df.n_inventadas > 0) & (df.rank_hyde_f > df.rank_dense_f)),
    ]

    for titulo, criterio, mask in casos:
        sel = df[mask]
        print(f"\n{'='*78}")
        print(f"  {titulo}")
        print(f"  criterio: {criterio}")
        print(f"  consultas que satisfazem: {len(sel)} de {len(df)}")
        print(f"{'='*78}")

        for _, row in sel.head(n_por_caso).iterrows():
            cd = row["cdacordao"]
            alvo = corpus_by_cd.get(cd, {})
            hd = hyde_docs.get(cd, {})
            ementa = alvo.get("ementa", "")
            hyde_txt = hd.get("hyde_ementa", "")
            consulta = alvo.get("descricao", "")

            print(f"\n  [idx {row['idx']}] processo {row['numero_processo']}")
            print(f"  classe: {row['classe']}")
            print(f"  cdacordao: {cd}\n")
            print("    Ranking do acordao-alvo:")
            print(_linha_ranking(row))

            c_alvo = citacoes(ementa)
            c_hyde = citacoes(hyde_txt)
            c_query = citacoes(consulta)
            t_alvo = termos(ementa)
            t_hyde = termos(hyde_txt)
            t_query = termos(consulta)

            acertos = sorted(c_hyde & c_alvo)
            inventados = sorted(c_hyde - c_alvo - c_query)
            compartilhados = sorted(t_hyde & t_alvo)

            print("\n    Dispositivos citados pelo HyDE que EXISTEM na ementa-alvo:")
            print(f"      {', '.join(acertos) if acertos else '(nenhum)'}")
            print("\n    Dispositivos citados pelo HyDE AUSENTES da ementa-alvo e da consulta:")
            print(f"      {', '.join(inventados) if inventados else '(nenhum)'}")
            print("\n    Termos de jargao compartilhados entre o doc do HyDE e a ementa:")
            print(f"      {', '.join(compartilhados[:20]) if compartilhados else '(nenhum)'}")

            # A consulta e o corpo integral (centenas de palavras); a ementa-alvo
            # e um header condensado. O HyDE reescreve a consulta NO FORMATO de
            # header, e e isso que aproxima os vetores. Jaccard mede esse efeito.
            j_query = jaccard(t_query, t_alvo)
            j_hyde = jaccard(t_hyde, t_alvo)
            print("\n    Similaridade lexical (Jaccard) com a ementa-alvo:")
            print(f"      consulta bruta x ementa : {j_query:.4f}  "
                  f"({len(t_query)} termos na consulta)")
            print(f"      doc do HyDE  x ementa   : {j_hyde:.4f}  "
                  f"({len(t_hyde)} termos no doc)")
            if j_query > 0:
                print(f"      razao HyDE/consulta     : {j_hyde / j_query:.2f}x")

            print(f"\n    Consulta (trecho):\n      {_amostra(consulta, 300)}")
            print(f"\n    Documento hipotetico do HyDE (trecho):\n      {_amostra(hyde_txt, 400)}")
            print(f"\n    Ementa-gabarito (trecho):\n      {_amostra(ementa, 300)}")
            print(f"\n  {'-'*74}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabela", action="store_true",
                    help="imprime so o resumo da tabela de rankings")
    ap.add_argument("--csv", help="exporta a tabela completa para CSV")
    ap.add_argument("--n", type=int, default=1,
                    help="quantos exemplos por caso")
    args = ap.parse_args()

    df = build_table()
    corpus_by_cd = {d["cdacordao"]: d for d in _load_json(CORPUS)}
    hyde_docs = load_hyde_docs()

    # Dispositivos que o HyDE citou mas que nao aparecem nem na ementa-alvo nem
    # na consulta original — a definicao operacional de alucinacao aqui.
    df["n_inventadas"] = [
        len(citacoes(hyde_docs.get(cd, {}).get("hyde_ementa", ""))
            - citacoes(corpus_by_cd.get(cd, {}).get("ementa", ""))
            - citacoes(corpus_by_cd.get(cd, {}).get("descricao", "")))
        for cd in df["cdacordao"]
    ]
    piorou = df.rank_hyde_f > df.rank_dense_f
    print(f"\n  Alucinacao de dispositivos legais em {int((df.n_inventadas > 0).sum())}"
          f"/{len(df)} consultas; destas, {int(((df.n_inventadas > 0) & piorou).sum())}"
          f" degradaram o ranking.")
    print(f"  Consultas em que o HyDE piorou vs. denso direto: {int(piorou.sum())}"
          f" — apenas {int(((df.n_inventadas > 0) & piorou).sum())} envolvem alucinacao.")

    if args.csv:
        df[["idx", "cdacordao", "numero_processo", "classe",
            *METODOS]].to_csv(args.csv, index=False, encoding="utf-8")
        print(f"tabela exportada para {args.csv}")

    print(f"\n  Tabela de rankings — {len(df)} consultas, recuperacao ate o top-10")
    print("  'fora do top-10' aparece como celula vazia (NaN), nao como rank > 10\n")
    for col in METODOS:
        achou = int((df[col + "_f"] <= 10).sum())
        topo = int((df[col + "_f"] == 1).sum())
        print(f"    {col:14s} no top-10: {achou:3d}/501   em 1o lugar: {topo:3d}")

    if args.tabela:
        return

    relatorio(df, corpus_by_cd, hyde_docs, n_por_caso=args.n)


if __name__ == "__main__":
    main()
