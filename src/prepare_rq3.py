"""
prepare_rq3.py
--------------
Extrai as 3 granularidades de consulta do campo "descricao" para o RQ3.

Cenarios:
  curto : apenas "I. CASO EM EXAME"
  medio : "I. CASO EM EXAME" + "II. QUESTAO EM DISCUSSAO"
  longo : descricao completa

Saida:
  data/queries/queries_rq3_curto.json
  data/queries/queries_rq3_medio.json
  data/queries/queries_rq3_longo.json

Variacoes cobertas pelo regex:
  "I. CASO EM EXAME", "I - CASO EM EXAME", "I - Caso em exame",
  "CASO EM EXAME" (sem numeral), "Caso em exame", etc.
  Fallback: se marcador ausente, usa texto completo (nenhum doc e perdido).

Método de Execução:
 ¬ python -m src.prepare_rq3
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.assets_v2 import CORPUS_V2_PATH

QUERIES_RQ3_DIR = Path("data/queries")
OUT_CURTO       = QUERIES_RQ3_DIR / "queries_rq3_curto.json"
OUT_MEDIO       = QUERIES_RQ3_DIR / "queries_rq3_medio.json"
OUT_LONGO       = QUERIES_RQ3_DIR / "queries_rq3_longo.json"

# Regex para cada secao — cobre numeral romano opcional + variacao de separador
# Numeral romano: I II III IV V VI VII VIII IX X XI XII XIII XIV
_ROMAN = r"(?:XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II|I)\s*[.\-–]\s*"

PAT_I   = re.compile(rf"(?:{_ROMAN})?CASO\s+EM\s+EXAME",          re.IGNORECASE)
PAT_II  = re.compile(rf"(?:{_ROMAN})?QUEST[AÃ]O\s+EM\s+DISCUSS",  re.IGNORECASE)
PAT_III = re.compile(rf"(?:{_ROMAN})?RAZ[OÕ]ES\s+DE\s+DECID",     re.IGNORECASE)


def _pos(text: str, pat: re.Pattern) -> int | None:
    m = pat.search(text)
    return m.start() if m else None


def extract_granularities(descricao: str) -> dict[str, str]:
    p1 = _pos(descricao, PAT_I)
    p2 = _pos(descricao, PAT_II)
    p3 = _pos(descricao, PAT_III)

    # Curto: secao I ate inicio da secao II (ou III se II ausente)
    if p1 is not None:
        end = p2 if p2 else p3
        curto = descricao[p1:end].strip() if end else descricao[p1:].strip()
    else:
        curto = descricao[:p2].strip() if p2 else descricao.strip()

    # Medio: do inicio de I (ou inicio do texto) ate inicio de III
    start_medio = p1 if p1 is not None else 0
    medio = descricao[start_medio:p3].strip() if p3 else descricao[start_medio:].strip()

    longo = descricao.strip()

    return {
        "curto": curto or descricao.strip(),
        "medio": medio or descricao.strip(),
        "longo": longo,
    }


def prepare_rq3_queries(corpus_path: str = CORPUS_V2_PATH) -> None:
    QUERIES_RQ3_DIR.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    n_docs = len(corpus)

    buckets = {"curto": [], "medio": [], "longo": []}
    fallbacks = {"curto": 0, "medio": 0}

    for doc in corpus:
        g = extract_granularities(doc["descricao"])
        base = {
            "cdacordao"      : doc["cdacordao"],
            "numero_processo": doc.get("numero_processo", ""),
            "classe"         : doc.get("classe", ""),
        }
        for granul in ("curto", "medio", "longo"):
            is_fallback = (g[granul] == doc["descricao"].strip())
            if granul != "longo" and is_fallback:
                fallbacks[granul] += 1
            buckets[granul].append({
                **base,
                "query"        : g[granul],
                "word_count"   : len(g[granul].split()),
                "granularidade": granul,
                "fallback"     : is_fallback if granul != "longo" else False,
            })

    for granul, path in [("curto", OUT_CURTO), ("medio", OUT_MEDIO), ("longo", OUT_LONGO)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(buckets[granul], f, ensure_ascii=False, indent=2)

    print(f"\n[prepare_rq3] {n_docs} documentos processados")
    print(f"  Extensao media (palavras):")
    for granul in ("curto", "medio", "longo"):
        media = sum(q["word_count"] for q in buckets[granul]) / n_docs
        print(f"    {granul:6s}: {media:.0f} palavras")
    print(f"  Fallbacks (marcador ausente -> texto completo usado):")
    for granul, count in fallbacks.items():
        print(f"    {granul:6s}: {count} docs")
    if fallbacks["curto"] > 0 or fallbacks["medio"] > 0:
        print(f"\n  Docs com fallback:")
        for q in buckets["curto"]:
            if q["fallback"]:
                print(f"    cdacordao={q['cdacordao']} | {q['query'][:70]}...")
    print(f"\n  Arquivos salvos:")
    for path in (OUT_CURTO, OUT_MEDIO, OUT_LONGO):
        print(f"    {path}")


if __name__ == "__main__":
    prepare_rq3_queries()