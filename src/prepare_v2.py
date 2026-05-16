"""
prepare_v2.py
-------------
Prepara os arquivos de corpus e queries para os experimentos v2.

Lê ementas_v2.json (campos: cdacordao, ementa, descricao, ...)
e gera:
  - data/corpus/ementas_v2.json  (ja existe, apenas valida)
  - data/queries/queries_v2.json (descricao como query, mesmo cdacordao)

No v2:
  corpus = ementa     (palavras-chave — o que indexamos no FAISS)
  query  = descricao  (texto completo do processo — o que buscamos)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.assets_v2 import CORPUS_V2_PATH, QUERIES_V2_PATH


def prepare(corpus_path: str = CORPUS_V2_PATH) -> None:
    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Valida campos obrigatorios
    required = {"cdacordao", "ementa", "descricao"}
    for i, doc in enumerate(docs):
        missing = required - doc.keys()
        if missing:
            raise ValueError(f"Documento {i} faltando campos: {missing}")

    print(f"[prepare_v2] Corpus carregado: {len(docs)} documentos")
    print(f"[prepare_v2] Exemplo ementa  : {docs[0]['ementa'][:80]}...")
    print(f"[prepare_v2] Exemplo descricao: {docs[0]['descricao'][:80]}...")

    # Gera queries_v2.json — descricao como query, mantendo cdacordao como chave
    queries = [
        {
            "cdacordao" : doc["cdacordao"],
            "numero_processo": doc.get("numero_processo", ""),
            "classe"    : doc.get("classe", ""),
            "query"     : doc["descricao"],   # query = descricao completa
        }
        for doc in docs
    ]

    Path(QUERIES_V2_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(QUERIES_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"[prepare_v2] queries_v2.json gerado: {len(queries)} queries em {QUERIES_V2_PATH}")


if __name__ == "__main__":
    prepare()