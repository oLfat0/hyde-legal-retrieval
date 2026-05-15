"""
hyde.py
-------
Geracao de N rodadas de documentos hipotetricos (HyDE) via vLLM Manager.

Fluxo:
  Para cada rodada n in 1..N_ROUNDS:
    Para cada query qi:
      hi_n = LLM_hyde(qi)   com T=0.7

  Saida: data/hyde_docs/hyde_docs_{n}.json  (n = 01, 02, ..., 10)

Cada arquivo espelha queries.json adicionando o campo "hyde_doc".
Rodadas sao independentes — o LLM recebe o mesmo prompt mas com T=0.7,
garantindo variabilidade entre documentos para o averaging de embeddings.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from src.assets import (
    VLLM_BASE_URL,
    DEFAULT_MODEL,
    LLM_AGENT_TIMEOUT,
    QUERIES_PATH,
)

# -- Configuracoes -------------------------------------------------------------
HYDE_DOCS_DIR   = Path("data/hyde_docs")
N_ROUNDS        = 15
HYDE_TEMPERATURE = 0.7   # variabilidade alta = melhor cobertura no averaging
HYDE_MAX_TOKENS  = 500
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5

HYDE_SYSTEM_PROMPT = (
    "Voce e um assistente juridico especializado em jurisprudencia brasileira. "
    "Sua tarefa e gerar uma ementa juridica hipotetica e plausivel a partir de um resumo. "
    "A ementa deve: (1) usar linguagem tecnico-juridica formal compativel com decisoes do TJMS; "
    "(2) incluir area do direito, tese juridica, dispositivos legais relevantes e resultado; "
    "(3) ter entre 300 e 500 palavras; "
    "(4) NAO inventar numeros de processo ou datas especificas; "
    "(5) responder APENAS com o texto da ementa, sem introducoes ou explicacoes."
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("VLLM_TOKEN")
    return OpenAI(base_url=VLLM_BASE_URL, api_key=api_key, timeout=LLM_AGENT_TIMEOUT)


def _hyde_path(n: int) -> Path:
    return HYDE_DOCS_DIR / f"hyde_docs_{n:02d}.json"


def _build_prompt(query: str) -> str:
    return (
        "Com base no resumo abaixo, gere uma ementa juridica hipotetica completa "
        "e tecnicamente precisa:\n\n"
        f"RESUMO:\n{query}\n\n"
        "EMENTA HIPOTETICA:"
    )


def _generate_one(query: str, client: OpenAI) -> str | None:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_prompt(query)},
                ],
                temperature=HYDE_TEMPERATURE,
                max_tokens=HYDE_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[hyde] tentativa {attempt}/{RETRY_ATTEMPTS} falhou: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    return None


def generate_round(n: int, resume: bool = True) -> list[dict]:
    """
    Gera a rodada n (1-based) de documentos hipotetricos.
    Salva em data/hyde_docs/hyde_docs_{n:02d}.json.

    Args:
        n      : numero da rodada (1 a N_ROUNDS)
        resume : pula registros ja gerados no arquivo de saida
    """
    HYDE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _hyde_path(n)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    existing: dict[str, str] = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        existing = {
            item["cdacordao"]: item["hyde_doc"]
            for item in saved
            if item.get("hyde_doc")
        }
        print(f"[hyde] rodada {n:02d}: retomando — {len(existing)}/{len(queries)} ja gerados")

    client  = _get_client()
    results = []
    failed  = 0

    for item in tqdm(queries, desc=f"Rodada {n:02d}", unit="doc"):
        cdac   = item["cdacordao"]
        record = {**item}

        if cdac in existing:
            record["hyde_doc"] = existing[cdac]
            results.append(record)
            continue

        doc = _generate_one(item["query"], client)
        if doc:
            record["hyde_doc"] = doc
        else:
            print(f"[hyde] FALHA cdacordao={cdac} rodada={n:02d}")
            record["hyde_doc"] = None
            failed += 1

        results.append(record)
        if len(results) % 10 == 0:
            _save(results, out_path)

    _save(results, out_path)
    print(f"[hyde] rodada {n:02d} concluida — {len(results)-failed}/{len(results)} ok")
    return results


def generate_all_rounds(resume: bool = True) -> None:
    """Gera todas as N_ROUNDS rodadas sequencialmente."""
    print(f"[hyde] Gerando {N_ROUNDS} rodadas de documentos hipotetricos (T={HYDE_TEMPERATURE})")
    for n in range(1, N_ROUNDS + 1):
        generate_round(n, resume=resume)
    print(f"[hyde] Todas as {N_ROUNDS} rodadas concluidas em {HYDE_DOCS_DIR}")


def _save(results: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_round(n: int) -> list[dict]:
    """
    Carrega rodada n do disco.

    Raises:
        FileNotFoundError : arquivo nao gerado ainda
        ValueError        : registros com hyde_doc=None
    """
    path = _hyde_path(n)
    if not path.exists():
        raise FileNotFoundError(
            f"Rodada {n:02d} nao encontrada: {path}\n"
            f"Execute: python -m src.hyde"
        )
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    failed = [d["cdacordao"] for d in docs if not d.get("hyde_doc")]
    if failed:
        raise ValueError(
            f"Rodada {n:02d}: {len(failed)} hyde_doc=None — "
            f"{failed[:3]}{'...' if len(failed) > 3 else ''}"
        )
    return docs


def load_all_rounds() -> list[list[dict]]:
    """Carrega todas as N_ROUNDS rodadas. Retorna lista de N_ROUNDS listas."""
    return [load_round(n) for n in range(1, N_ROUNDS + 1)]


if __name__ == "__main__":
    print(f"[hyde] Iniciando HyDE timer...")
    start = time.time()
    generate_all_rounds(resume=True)
    end = time.time()
    tempo = end-start
    print(f"[hyde] Tempo de Geração: {(tempo)/3600:.2f}h ({(tempo)/60:.2f}min | {tempo:.2f}s)")