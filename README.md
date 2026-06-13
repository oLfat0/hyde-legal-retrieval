# HyDE for Legal Retrieval (Brazilian Jurisprudence)

Undergraduate Research Project — Information Retrieval & NLP

This repository contains the datasets and experimental pipeline developed for an undergraduate research project investigating the impact of **Hypothetical Document Embeddings (HyDE)** on the retrieval of Brazilian legal precedents.

The study evaluates whether semantic expansion of queries using large language models improves dense retrieval performance on a corpus of judicial decisions from the Brazilian legal system.

---

## Research Question

**Does the HyDE technique improve semantic retrieval of legal precedents in Brazilian jurisprudence?**

---

## Overview

Recent advances in **dense retrieval** have demonstrated strong performance in semantic search tasks by representing queries and documents as vectors in a shared embedding space. However, short queries often lack sufficient semantic information for accurate representation.

The **HyDE (Hypothetical Document Embeddings)** technique addresses this issue by generating a hypothetical document from the query using a language model. The embedding of this generated document is then used for retrieval instead of the original query.

This project evaluates the effectiveness of this technique in a **Brazilian legal domain**, using a corpus of judicial decision summaries (*ementas*).

---

## Experimental Setup

**Corpus**

Approximately **500 public legal decisions** extracted from the jurisprudence system of the Mato Grosso do Sul Court of Justice (TJMS).

**Query Generation**

Synthetic queries were generated automatically using **LLM-based summaries** of each document.

**Retrieval Strategies Compared**

1. **Baseline** – embedding of the generated summary  
2. **HyDE** – embedding of a hypothetical document generated from the summary

Document embeddings were indexed using **FAISS** and evaluated using standard information retrieval metrics.

---

## Evaluation Metrics

The following metrics were used:

- **Recall@k**
- **Mean Reciprocal Rank (MRR)**
- **nDCG@k**

A controlled **self-retrieval protocol** was adopted, where each query is expected to retrieve its original document.

---

## Repository Structure
```
├── data/           
│ ├── corpus/           # legal decision summaries (ementas)
│ └── queries/          # generated mini-summaries and HyDE documents
├── src/                # experimental pipeline and retrieval scripts
├── experiments/        # experiment configurations
└── results/            # evaluation results and metric outputs
```     


---

## Contributions

- Benchmark for **dense retrieval in Brazilian legal texts**
- Empirical evaluation of **HyDE in Portuguese legal documents**
- Reproducible experimental pipeline for **legal information retrieval**

---

## Research Context

This repository is part of an **undergraduate research project in Computer Engineering**, focusing on **Information Retrieval and Natural Language Processing applied to the legal domain**.

The goal is to investigate the applicability of modern retrieval techniques in Brazilian legal datasets and contribute to the development of reproducible benchmarks for the area.

## Running the Experiments

### Pipeline Order

The full RQ3 pipeline (query-granularity analysis) runs in the following order:

#### 1. Prepare queries by granularity

Extracts the three granularity levels (`curto`, `medio`, `longo`) from each document's structured body:

```bash
python -m src.prepare_rq3
```

This generates:
- `data/queries/queries_rq3_curto.json` — section "I. CASO EM EXAME" only
- `data/queries/queries_rq3_medio.json` — sections I + II
- `data/queries/queries_rq3_longo.json` — full structured body

#### 2. Generate hypothetical documents (HyDE) per granularity

Runs the LLM (Gemma 3 via OpenRouter) to generate 5 rounds of hypothetical ementas for each scenario:

```bash
python -m src.hyde_rq3 --granul curto
python -m src.hyde_rq3 --granul medio
python -m src.hyde_rq3 --granul longo
```

#### 3. Compute embeddings and averaged vectors

Embeds each round and computes the L2-normalized mean across the 5 rounds:

```bash
python -m src.hyde_embedder_rq3 --granul curto
python -m src.hyde_embedder_rq3 --granul medio
python -m src.hyde_embedder_rq3 --granul longo
```

#### 4. Run the retrieval experiments

For each scenario, run the four configurations. The FAISS and BM25 indexes are built automatically on first run (full corpus):

```bash
# CURTO
python experiments_rq3/curto/baseline_dense_curto.py
python experiments_rq3/curto/baseline_hybrid_curto.py
python experiments_rq3/curto/hyde_dense_curto.py
python experiments_rq3/curto/hyde_hybrid_curto.py

# MEDIO
python experiments_rq3/medio/baseline_dense_medio.py
python experiments_rq3/medio/baseline_hybrid_medio.py
python experiments_rq3/medio/hyde_dense_medio.py
python experiments_rq3/medio/hyde_hybrid_medio.py

# LONGO
python experiments_rq3/longo/baseline_dense_longo.py
python experiments_rq3/longo/baseline_hybrid_longo.py
python experiments_rq3/longo/hyde_dense_longo.py
python experiments_rq3/longo/hyde_hybrid_longo.py
```

Results are saved as JSON files in `results_rq3/`.

### Notes

- The `longo` scenario uses the complete structured body and is equivalent to the main RQ1/RQ2 experiments — if those results already exist, they can be reused.
- Indexes are cached on disk. To force a rebuild (e.g. after changing the corpus), delete the relevant folders under `data/` (`faiss_index_v2/`, `bm25_index_v2/`) and the `data/hyde_rq3_docs/` and `data/hyde_rq3_embeds/` caches.
- The baseline configurations do not depend on the LLM — only the HyDE configurations invoke the language model.

---

## License

This repository uses only **publicly available legal documents**. Please refer to the respective court systems for the original sources.