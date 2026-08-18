"""
corpus_stats.py — Levanta estatisticas descritivas do corpus (ementas_v2.json)
para a secao de Metodologia do artigo.

Campos do dataset:
  - ementa     -> "Header"  (palavras-chave / headnote indexado no FAISS)
  - descricao  -> "Body"    (corpo estruturado do acordao, usado como query)
  - classe     -> classe processual + assunto ("Classe / Assunto")
  - ementa[0]  -> ramo do direito ("DIREITO X.")

Saida: data/corpus/corpus_stats.json
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/corpus/ementas_v2.json")
OUT_PATH    = Path("data/corpus/corpus_stats.json")


def words(text: str) -> int:
    return len(text.split())


def ano_processo(numero: str):
    # numero CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO  -> ano = grupo AAAA
    m = re.search(r"\.(\d{4})\.", numero)
    return int(m.group(1)) if m else None


def classe_processual(classe: str) -> str:
    # parte antes da primeira "/" = classe processual pura (sem o assunto)
    return classe.split("/")[0].strip()


def ramo_composto(ementa: str) -> str:
    # ementas no padrao CNJ trazem um cabecalho "DIREITO <RAMO> [E <RAMO>...]".
    # O cabecalho nem sempre abre a ementa (as vezes vem apos a classe), por
    # isso procuramos o token "DIREITO" em qualquer ponto e capturamos ate o
    # proximo terminador ("." ou travessao "-", "–", "—").
    m = re.search(r"\bD?IREITO\b(.*?)(?=[.\-–—])", ementa.strip(), flags=re.IGNORECASE)
    if not m:
        return "Indefinido"
    ramo = re.sub(r"\s+", " ", "DIREITO" + m.group(1)).strip()
    return ramo.title()


def ramos_individuais(composto: str):
    # quebra "Direito Penal E Processual Penal" -> ["Penal", "Processual Penal"]
    if composto == "Indefinido":
        return []
    pecas = re.split(r"\s+e\s+", composto, flags=re.IGNORECASE)
    ramos = []
    for p in pecas:
        # remove eventual "Direito " que se repete em cada ramo do cabecalho
        r = re.sub(r"^\s*direito\s+", "", p, flags=re.IGNORECASE).strip()
        if r:
            ramos.append(r.title())
    return ramos


def main():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    n = len(corpus)

    header_words = [words(d["ementa"])    for d in corpus]
    body_words   = [words(d["descricao"]) for d in corpus]

    anos        = [a for d in corpus if (a := ano_processo(d["numero_processo"]))]
    classes     = [classe_processual(d["classe"]) for d in corpus]
    compostos   = [ramo_composto(d["ementa"])     for d in corpus]

    classe_freq    = Counter(classes)
    composto_freq  = Counter(compostos)

    # cada decisao conta para cada ramo individual que seu cabecalho menciona
    individual_freq = Counter()
    for c in compostos:
        individual_freq.update(ramos_individuais(c))
    ramo_top, ramo_top_n = individual_freq.most_common(1)[0]
    n_indefinido = composto_freq.get("Indefinido", 0)

    stats = {
        "corpus": {
            "fonte": str(CORPUS_PATH).replace("\\", "/"),
            "total_decisoes": n,
            "periodo": {"ano_inicial": min(anos), "ano_final": max(anos)},
        },
        "header_ementa": {
            "descricao": "Palavras-chave / headnote (campo 'ementa', indexado no FAISS)",
            "tamanho_medio_palavras": round(statistics.mean(header_words), 1),
            "mediana_palavras": int(statistics.median(header_words)),
            "minimo_palavras": min(header_words),
            "maximo_palavras": max(header_words),
        },
        "body_descricao": {
            "descricao": "Corpo estruturado do acordao (campo 'descricao', usado como query)",
            "tamanho_medio_palavras": round(statistics.mean(body_words), 1),
            "mediana_palavras": int(statistics.median(body_words)),
            "minimo_palavras": min(body_words),
            "maximo_palavras": max(body_words),
        },
        "classes_processuais": {
            "quantidade_distintas": len(classe_freq),
            "top_5": dict(classe_freq.most_common(5)),
        },
        "areas_do_direito": {
            "descricao": "Ramo do direito extraido do cabecalho 'DIREITO ...' da ementa",
            "area_predominante": ramo_top,
            "area_predominante_n": ramo_top_n,
            "area_predominante_pct": round(100 * ramo_top_n / n, 1),
            "ramos_individuais": {
                "quantidade_distintas": len(individual_freq),
                "ranking": dict(individual_freq.most_common()),
            },
            "ramos_compostos": {
                "quantidade_distintas": len([k for k in composto_freq if k != "Indefinido"]),
                "indefinido_n": n_indefinido,
                "top_5": dict(composto_freq.most_common(5)),
            },
        },
    }

    OUT_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[corpus_stats] {n} decisoes processadas -> {OUT_PATH}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
