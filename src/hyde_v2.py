"""
hyde_v2.py
----------
Geracao de ementas hipotéticas (HyDE v2) a partir da descricao do processo.

Diferenca fundamental em relacao ao hyde.py original:
  - Input  : descricao completa do processo (query)
  - Output : ementa hipotetica (sequencia de palavras-chave juridicas)
             no formato exato das ementas do TJMS

O prompt instrui o modelo a gerar APENAS a linha de palavras-chave
no estilo das ementas reais, nao um texto descritivo longo.

Gera HYDE_N_ROUNDS=5 rodadas independentes (T=0.7).
Saida: data/hyde_v2_docs/hyde_v2_docs_{01..05}.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from src.assets_v2 import (
    VLLM_BASE_URL,
    DEFAULT_MODEL,
    LLM_AGENT_TIMEOUT,
    HYDE_TEMPERATURE,
    HYDE_MAX_TOKENS,
    HYDE_N_ROUNDS,
    QUERIES_V2_PATH,
    HYDE_V2_DOCS_DIR,
)

RETRY_ATTEMPTS = 3
RETRY_DELAY    = 5

# -- Prompt HyDE v2 ------------------------------------------------------------
# Objetivo: gerar uma EMENTA (sequência de palavras-chave) a partir da descricao do processo.
# O modelo deve imitar o formato compacto das ementas do TJMS:
#   "AREA DO DIREITO. TEMA PRINCIPAL. SUBTEMA. RESULTADO."

HYDE_SYSTEM_PROMPT_V2 = (
    "Voce e um especialista em jurisprudencia do Tribunal de Justica de Mato Grosso do Sul (TJMS). "
    "Sua tarefa e gerar a EMENTA de uma decisao judicial a partir da descricao do processo. "
    "A ementa deve seguir EXATAMENTE o formato das ementas do TJMS: "
    "uma sequencia de PALAVRAS-CHAVE e TEMAS em maiusculas, separados por pontos, "
    "sem texto corrido, sem introducoes, sem numeracao. "
    "Exemplo de formato: "
    "'DIREITO PROCESSUAL CIVIL. AGRAVO INTERNO. ADMISSIBILIDADE. TEMA 339 DO STF. "
    "FUNDAMENTACAO SUFICIENTE. ERRO GROSSEIRO. RECURSO DESPROVIDO.' "
    "Responda APENAS com a ementa, sem nenhuma explicacao adicional."
)

def _build_prompt(descricao: str) -> str:
    return (
        "Com base na descricao do processo abaixo, gere a ementa correspondente "
        "no formato de palavras-chave do TJMS:\n\n"
        f"DESCRICAO:\n{descricao}\n\n"
        "EMENTA:"
    )


def _get_client() -> OpenAI:
    api_key = os.environ.get("VLLM_TOKEN")
    return OpenAI(base_url=VLLM_BASE_URL, api_key=api_key, timeout=LLM_AGENT_TIMEOUT)


def _generate_one(descricao: str, client: OpenAI) -> str | None:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT_V2},
                    {"role": "user",   "content": _build_prompt(descricao)},
                ],
                temperature=HYDE_TEMPERATURE,
                max_tokens=HYDE_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[hyde_v2] tentativa {attempt}/{RETRY_ATTEMPTS} falhou: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    return None


def _hyde_path(n: int) -> Path:
    return Path(HYDE_V2_DOCS_DIR) / f"hyde_v2_docs_{n:02d}.json"


def _save(results: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_round(n: int, resume: bool = True) -> list[dict]:
    """
    Gera a rodada n de ementas hipotéticas.
    Cada registro contem o cdacordao + a ementa hipotética gerada.
    """
    Path(HYDE_V2_DOCS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = _hyde_path(n)

    with open(QUERIES_V2_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    existing: dict[str, str] = {}
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        existing = {
            item["cdacordao"]: item["hyde_ementa"]
            for item in saved
            if item.get("hyde_ementa")
        }
        print(f"[hyde_v2] rodada {n:02d}: retomando — {len(existing)}/{len(queries)} prontos")

    client  = _get_client()
    results = []
    failed  = 0

    for item in tqdm(queries, desc=f"HyDE v2 rodada {n:02d}", unit="doc"):
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
            print(f"[hyde_v2] FALHA cdacordao={cdac} rodada={n:02d}")
            record["hyde_ementa"] = None
            failed += 1

        results.append(record)
        if len(results) % 10 == 0:
            _save(results, out_path)

    _save(results, out_path)
    print(f"[hyde_v2] rodada {n:02d} concluida — {len(results)-failed}/{len(results)} ok")
    return results


def generate_all_rounds(resume: bool = True) -> None:
    print(f"[hyde_v2] Gerando {HYDE_N_ROUNDS} rodadas de ementas hipotéticas (T={HYDE_TEMPERATURE})")
    for n in range(1, HYDE_N_ROUNDS + 1):
        generate_round(n, resume=resume)
    print(f"[hyde_v2] Concluido — arquivos em {HYDE_V2_DOCS_DIR}")


def load_round(n: int) -> list[dict]:
    path = _hyde_path(n)
    if not path.exists():
        raise FileNotFoundError(
            f"Rodada {n:02d} nao encontrada: {path}\n"
            "Execute: python -m src.hyde_v2"
        )
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    failed = [d["cdacordao"] for d in docs if not d.get("hyde_ementa")]
    if failed:
        raise ValueError(f"Rodada {n:02d}: {len(failed)} hyde_ementa=None — {failed[:3]}")
    return docs


def load_all_rounds() -> list[list[dict]]:
    return [load_round(n) for n in range(1, HYDE_N_ROUNDS + 1)]


if __name__ == "__main__":
    generate_all_rounds(resume=True)