"""
hyde_case_studies.py — Extrai um caso de SUCESSO e um de FALHA do HyDE para
analise qualitativa (tabela: Query | HyDE Generated Header | Real Header | Rank).

Alinhamento (verificado): para todo indice i,
    corpus[i]  ==  hyde_v2_docs_##[i]  ==  per_query[i]  (mesmo cdacordao, N=501)
Logo per_query[i].rank e a posicao (1-based; null = fora do top-10) do doc
relevante da query i em cada metodo.

Casos escolhidos:
  SUCESSO  idx 87  — HyDE chega ao topo (rank 1); baseline_dense, baseline_hybrid
                     e BM25 nao recuperam o doc no top-10.
  FALHA    idx 132 — HyDE perde o doc (rank null); todos os demais metodos acertam
                     em rank 1. As 5 geracoes alucinam o enquadramento juridico.

Saida: results_v2/hyde_case_studies.json
"""

import json
from pathlib import Path

CORPUS_PATH = Path("data/corpus/ementas_v2.json")
DOCS_GLOB   = "data/hyde_v2_docs/hyde_v2_docs_{n:02d}.json"
RESULTS_DIR = Path("results_v2")
OUT_PATH    = RESULTS_DIR / "hyde_case_studies.json"

METHODS = {
    "hyde_dense_v2":     "HyDE + Dense (metodo do estudo)",
    "hyde_hybrid_v2":    "HyDE + Hibrido (dense + BM25)",
    "baseline_dense_v2": "Baseline Dense (descricao crua como query)",
    "baseline_hybrid_v2":"Baseline Hibrido",
    "bm25_puro_v2":      "BM25 puro (lexico)",
}

# indices selecionados manualmente apos inspecao qualitativa
SUCCESS_IDX = 87
FAILURE_IDX = 132


def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    corpus = load_json(CORPUS_PATH)
    docs   = [load_json(DOCS_GLOB.format(n=n)) for n in range(1, 6)]
    per_q  = {m: load_json(RESULTS_DIR / f"{m}.json")["per_query"] for m in METHODS}

    # sanity-check de alinhamento
    cd0 = [c["cdacordao"] for c in corpus]
    for d in docs:
        assert [x["cdacordao"] for x in d] == cd0, "ordem docs != corpus"
    for m in METHODS:
        assert len(per_q[m]) == len(corpus), f"per_query {m} tamanho != corpus"

    def build_case(idx, rotulo, descricao_caso):
        c = corpus[idx]
        geracoes = [docs[n][idx]["hyde_ementa"] for n in range(5)]
        ranks = {m: per_q[m][idx]["rank"] for m in METHODS}
        body = c["descricao"]
        return {
            "rotulo": rotulo,
            "descricao_do_caso": descricao_caso,
            "indice_no_corpus": idx,
            "numero_processo": c["numero_processo"],
            "cdacordao": c["cdacordao"],
            "classe": c["classe"],
            "word_count": c["word_count"],
            # --- colunas pedidas para a tabela ---
            "tabela": {
                "query_trecho_corpo": body[:400] + ("..." if len(body) > 400 else ""),
                "hyde_generated_header": geracoes[0],   # rodada 1 (representativa)
                "real_header": c["ementa"],
                "rank_hyde_dense": ranks["hyde_dense_v2"],
            },
            # --- conteudo integral para analise ---
            "real_header_ementa": c["ementa"],
            "body_descricao": body,
            "hyde_geracoes_5_rodadas": geracoes,
            "nota_recuperacao": (
                "A recuperacao HyDE usa a MEDIA L2-normalizada dos embeddings das 5 "
                "geracoes; nenhum texto isolado e 'a' query. As 5 geracoes estao "
                "listadas em hyde_geracoes_5_rodadas."
            ),
            "ranks_por_metodo": {
                m: {"rank": ranks[m], "metodo": METHODS[m]} for m in METHODS
            },
        }

    out = {
        "descricao": (
            "Estudos de caso qualitativos do HyDE (N=501). 'rank' e a posicao 1-based "
            "do acordao-alvo no ranking de cada metodo; null = fora do top-10."
        ),
        "fontes": {
            "corpus": str(CORPUS_PATH).replace("\\", "/"),
            "hyde_docs": "data/hyde_v2_docs/hyde_v2_docs_01..05.json",
            "resultados": [str(RESULTS_DIR / f"{m}.json").replace("\\", "/") for m in METHODS],
        },
        "caso_sucesso": build_case(
            SUCCESS_IDX, "SUCESSO",
            "HyDE expandiu a descricao numa ementa hipotetica que capturou termos "
            "distintivos do header real (Execucao Fiscal/IPTU/Resolucao CNJ 547-2024/"
            "Tema 1.184 STF), levando o acordao ao TOPO (rank 1). Os demais metodos "
            "(baseline dense, baseline hibrido e BM25) nao recuperaram o doc no top-10.",
        ),
        "caso_falha": build_case(
            FAILURE_IDX, "FALHA",
            "HyDE 'se perdeu': as 5 geracoes alucinaram o enquadramento juridico "
            "(abrindo com 'FINS E INVESTIMENTOS'/'FINANCAS', termo em espanhol "
            "'CADENA DE', e ruido 'JURUS/JURISTAS DE MORA'), desviando de uma acao "
            "de descontos indevidos em beneficio previdenciario / relacao de consumo. "
            "O acordao caiu fora do top-10 no HyDE, enquanto baseline dense, baseline "
            "hibrido e BM25 acertaram em rank 1.",
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hyde_case_studies] gravado -> {OUT_PATH}")
    for caso in (out["caso_sucesso"], out["caso_falha"]):
        ranks_str = ", ".join(f"{k}:{v['rank']}" for k, v in caso["ranks_por_metodo"].items())
        print(f"  {caso['rotulo']:8s} proc={caso['numero_processo']} "
              f"rank_hyde={caso['tabela']['rank_hyde_dense']} ranks={{ {ranks_str} }}")


if __name__ == "__main__":
    main()
