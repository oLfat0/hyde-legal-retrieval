"""
hyde_rq3.py
-----------
Gera documentos hipotetricos para os 3 cenarios do RQ3.
Reutiliza o mesmo LLM e prompt do hyde_v2.py — so muda o arquivo de queries.

Uso:
  python -m src.hyde_rq3 --granul curto
  python -m src.hyde_rq3 --granul medio
  python -m src.hyde_rq3 --granul longo
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from src.assets_v2 import (
    VLLM_BASE_URL, DEFAULT_MODEL, LLM_AGENT_TIMEOUT,
    HYDE_TEMPERATURE, HYDE_MAX_TOKENS, HYDE_N_ROUNDS,
)
from src.hyde_v2 import HYDE_SYSTEM_PROMPT_V2, _build_prompt, _generate_one
from dotenv import load_dotenv

load_dotenv()

HYDE_RQ3_DIR = Path("data/hyde_rq3_docs")
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5


def _hyde_path(granul: str, n: int) -> Path:
    return HYDE_RQ3_DIR / granul / f"hyde_rq3_{granul}_{n:02d}.json"


def _save(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _get_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY")
    return OpenAI(base_url=VLLM_BASE_URL, api_key=api_key, timeout=LLM_AGENT_TIMEOUT)


def generate_round(granul: str, n: int, resume: bool = True) -> list[dict]:
    """Gera rodada n para o cenario granul."""
    queries_path = Path(f"data/queries/queries_rq3_{granul}.json")
    out_path     = _hyde_path(granul, n)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    existing: dict[str, str] = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        existing = {item["cdacordao"]: item["hyde_ementa"]
                    for item in saved if item.get("hyde_ementa")}
        print(f"[hyde_rq3] {granul} rodada {n:02d}: retomando — {len(existing)}/{len(queries)} prontos")

    from tqdm import tqdm
    client  = _get_client()
    results = []
    failed  = 0

    for item in tqdm(queries, desc=f"HyDE RQ3 {granul} r{n:02d}", unit="doc"):
        cdac   = item["cdacordao"]
        record = {**item}

        if cdac in existing:
            record["hyde_ementa"] = existing[cdac]
            results.append(record)
            continue

        hyde_ementa = _generate_one(item["query"], client)
        if hyde_ementa:
            record["hyde_ementa"] = hyde_ementa
        else:
            print(f"[hyde_rq3] FALHA cdacordao={cdac}")
            record["hyde_ementa"] = None
            failed += 1

        results.append(record)
        if len(results) % 10 == 0:
            _save(results, out_path)

    _save(results, out_path)
    print(f"[hyde_rq3] {granul} rodada {n:02d}: {len(results)-failed}/{len(results)} ok")
    return results


def generate_all_rounds(granul: str, resume: bool = True) -> None:
    print(f"[hyde_rq3] Gerando {HYDE_N_ROUNDS} rodadas — cenario: {granul}")
    for n in range(1, HYDE_N_ROUNDS + 1):
        generate_round(granul, n, resume=resume)
    print(f"[hyde_rq3] {granul} concluido")


def load_round(granul: str, n: int) -> list[dict]:
    path = _hyde_path(granul, n)
    if not path.exists():
        raise FileNotFoundError(f"Rodada {n:02d} do cenario '{granul}' nao encontrada: {path}")
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    failed = [d["cdacordao"] for d in docs if not d.get("hyde_ementa")]
    if failed:
        raise ValueError(f"Cenario {granul} rodada {n:02d}: {len(failed)} hyde_ementa=None")
    return docs


if __name__ == "__main__":
    import time
    start = time.time()
    if start:
        print("Timer Iniciado...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--granul", choices=["curto","medio","longo"], required=True)
    args = parser.parse_args()
    generate_all_rounds(args.granul, resume=True)
    end = time.time()
    print(f"Tempo: {(end-start)/3600:.2f}h ({(end-start)/60:.2f}min)")