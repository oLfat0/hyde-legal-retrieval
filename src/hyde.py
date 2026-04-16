"""
hyde.py
-------
Geracao de documentos hipotetricos (HyDE) via vLLM Manager (Gemma-3-12b-it).

Fluxo:
  1. Le as queries de data/queries/queries.json
  2. Para cada query qi, chama o InstructGPT(Gemma-12b) para gerar documento juridico hipotetico hi
  3. Salva em data/hyde_docs/hyde_docs.json (pre-gerado e estatico = reprodutibilidade)

O arquivo espelha a estrutura do queries.json adicionando o campo "hyde_doc".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from src.assets import (
    VLLM_BASE_URL,
    DEFAULT_MODEL,
    LLM_AGENT_TIMEOUT,
    LLM_TEMPERATURE,
    QUERIES_PATH,
)

# -- Configuracoes HyDE --------------------------------------------------------
HYDE_DOCS_DIR   = Path("data/hyde_docs")
HYDE_DOCS_PATH  = HYDE_DOCS_DIR / "hyde_docs.json"
HYDE_MAX_TOKENS = 500
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5   # segundos entre tentativas

# -- Cliente vLLM --------------------------------------------------------------

def _get_client() -> OpenAI:
    import os
    api_key = os.environ.get("VLLM_TOKEN")
    return OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=api_key,
        timeout=LLM_AGENT_TIMEOUT,
    )

# -- Prompt HyDE ---------------------------------------------------------------

HYDE_SYSTEM_PROMPT = (
    "Voce e um assistente juridico especializado em jurisprudencia brasileira. "
    "Sua tarefa e gerar uma ementa juridica hipotetica e plausivel a partir de um resumo. "
    "A ementa deve: (1) usar linguagem tecnico-juridica formal compativel com decisoes do TJMS; "
    "(2) incluir area do direito, tese juridica, dispositivos legais relevantes e resultado; "
    "(3) ter entre 300 e 500 palavras; "
    "(4) NAO inventar numeros de processo ou datas especificas; "
    "(5) responder APENAS com o texto da ementa, sem introducoes ou explicacoes."
)

def _build_hyde_prompt(query: str) -> str:
    return (
        "Com base no resumo abaixo, gere uma ementa juridica hipotetica completa "
        "e tecnicamente precisa:\n\n"
        f"RESUMO:\n{query}\n\n"
        "EMENTA HIPOTETICA:"
    )

# -- Geracao de um documento (com retry) ---------------------------------------

def generate_hyde_doc(query: str, client: OpenAI) -> str | None:
    """
    Gera documento hipotetico hi a partir da query qi.
    Retorna o texto ou None apos todas as tentativas falharem.
    """
    prompt = _build_hyde_prompt(query)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=HYDE_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[hyde] Tentativa {attempt}/{RETRY_ATTEMPTS} falhou: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)

    return None

# -- Geracao em lote com checkpoint --------------------------------------------

def generate_all_hyde_docs(resume: bool = True) -> list[dict]:
    """
    Gera documentos hipotetricos para todas as queries.

    Args:
        resume : se True, pula queries ja geradas no arquivo de saida
                 (permite retomar apos interrupcao sem reprocessar)

    Retorna lista de dicts com todos os campos originais + "hyde_doc".
    """
    HYDE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Carrega progresso anterior
    existing: dict[str, str] = {}
    if resume and HYDE_DOCS_PATH.exists():
        with open(HYDE_DOCS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        existing = {
            item["cdacordao"]: item["hyde_doc"]
            for item in saved
            if item.get("hyde_doc")
        }
        print(f"[hyde] Retomando: {len(existing)}/{len(queries)} ja gerados")

    client  = _get_client()
    results = []
    failed  = 0

    for item in tqdm(queries, desc="Gerando docs HyDE", unit="doc"):
        cdacordao = item["cdacordao"]
        record    = {**item}

        if cdacordao in existing:
            record["hyde_doc"] = existing[cdacordao]
            results.append(record)
            continue

        hyde_doc = generate_hyde_doc(item["query"], client)

        if hyde_doc:
            record["hyde_doc"] = hyde_doc
        else:
            print(f"[hyde] FALHA cdacordao={cdacordao} — registrado como null")
            record["hyde_doc"] = None
            failed += 1

        results.append(record)

        # Checkpoint a cada 10 documentos
        if len(results) % 10 == 0:
            _save(results)

    _save(results)

    total   = len(results)
    success = total - failed
    print(f"[hyde] Concluido: {success}/{total} gerados com sucesso")
    if failed:
        print(f"[hyde] ATENCAO: {failed} falhas — verifique registros hyde_doc=null")

    return results

def generate_ONE_hyde_docs(resume: bool = True) -> list[dict]:
    """
    Gera UM documento hipotetrico para determinada query.

    Args:
        resume : se True, pula queries ja geradas no arquivo de saida
                 (permite retomar apos interrupcao sem reprocessar)

    Retorna lista de dicts com todos os campos originais + "hyde_doc".
    """
    HYDE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    client  = _get_client()
    results = []
    failed  = 0
    
    item = queries[0]

    cdacordao = item["cdacordao"]
    proc_id = item["numero_processo"]
    record    = {**item}

    hyde_doc = generate_hyde_doc(item["query"], client)

    if hyde_doc:
        record["hyde_doc"] = hyde_doc
    else:
        print(f"[hyde] FALHA cdacordao={cdacordao} — registrado como null")
        record["hyde_doc"] = None
        failed += 1

    results.append(record)

    _save(results)

    print(f"[hyde] Concluido: Processo  {proc_id} gerado com sucesso")
    if failed:
        print(f"[hyde] ATENCAO: {failed} falhas — verifique registros hyde_doc=null")

    return results

def _save(results: list[dict]) -> None:
    with open(HYDE_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_hyde_docs() -> list[dict]:
    """
    Carrega documentos hipotéticos pre-gerados do disco.

    Raises:
        FileNotFoundError : arquivo nao encontrado (rode generate_all_hyde_docs primeiro)
        ValueError        : registros com hyde_doc=None encontrados
    """
    if not HYDE_DOCS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo HyDE nao encontrado: {HYDE_DOCS_PATH}\n"
            "Execute: python -m src.hyde"
        )

    with open(HYDE_DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    failed = [d["cdacordao"] for d in docs if not d.get("hyde_doc")]
    if failed:
        raise ValueError(
            f"{len(failed)} documentos com hyde_doc=None: {failed[:5]}"
            f"{'...' if len(failed) > 5 else ''}\n"
            "Regere os documentos faltantes antes de prosseguir."
        )

    return docs


if __name__ == "__main__":
    print("=== Geracao de documentos HyDE ===")
    docs = generate_all_hyde_docs(resume=True)
    print(f"Salvo em {HYDE_DOCS_PATH}: {len(docs)} documentos")