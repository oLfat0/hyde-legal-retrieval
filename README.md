# HyDE for Legal Retrieval (Brazilian Jurisprudence)

Undergraduate Research Project — Information Retrieval & NLP

This repository contains the dataset and experimental pipeline developed for an undergraduate research project investigating the impact of **Hypothetical Document Embeddings (HyDE)** on the retrieval of Brazilian legal precedents.

The study evaluates whether semantic expansion of queries via large language models improves dense and hybrid retrieval performance on a corpus of structured judicial decisions (*ementas*) from the Court of Justice of Mato Grosso do Sul (TJMS).

---

## Research Questions

The work is guided by three research questions:

- **RQ1 (General Effectiveness):** Does HyDE-based semantic expansion improve retrieval over traditional lexical (BM25) and pure dense baselines?
- **RQ2 (Hybridization):** Does combining HyDE with hybrid retrieval (dense + sparse via RRF) outperform the isolated approaches?
- **RQ3 (Query Granularity):** How does the length and specificity of the query (isolated factual sections vs. the full structured body) affect the quality of the HyDE-generated document?

---

## Overview

**Dense retrieval** represents queries and documents as vectors in a shared embedding space, capturing semantic similarity beyond exact lexical matching. However, queries phrased very differently from the target documents suffer from vocabulary mismatch.

The **HyDE (Hypothetical Document Embeddings)** technique mitigates this by prompting a language model to generate a hypothetical document from the query, then using that document's embedding for retrieval. This project evaluates the technique in the **Brazilian legal domain**, where the dense, jargon-heavy language of court rulings poses a strong vocabulary-mismatch challenge.

---

## Experimental Setup

**Corpus**

501 public legal decisions extracted from the TJMS jurisprudence system, focusing on civil-law rulings. Each decision is natively split by the court into two components:

- **Cabeçalho da Ementa** (header — indexers and legal thesis): acts as the retrieval **key** (document-alvo).
- **Corpo Estruturado da Ementa** (structured body — facts, controversy, reasoning): acts as the **query**.

**Models**

- Embedding: `facebook/mcontriever-msmarco` (768-dim), zero-shot, no fine-tuning.
- HyDE generation: `gemma-3-12b-it`, temperature 0.7, 5 generation rounds averaged.

**Retrieval strategies compared**

1. **Baseline + Dense** — embedding of the original query (structured body).
2. **HyDE + Dense** — embedding of the averaged hypothetical headers.
3. **Baseline + Hybrid** — dense + BM25 fused via Reciprocal Rank Fusion (RRF).
4. **HyDE + Hybrid** — HyDE dense + BM25 fused via RRF.

Target embeddings are indexed statically with **FAISS** (`IndexFlatIP` over L2-normalized vectors, equivalent to cosine similarity).

---

## Evaluation Metrics

A controlled **auto-retrieval** protocol is adopted: each query has exactly one relevant document (its own header). Performance is measured with:

- **Recall@k** (k = 1, 5, 10)
- **Mean Reciprocal Rank (MRR)**
- **nDCG@k**

---

## Repository Structure

```
├── data/
│   ├── corpus/              # structured ementas (ementas_v2.json)
│   ├── queries/             # queries per granularity (rq3) and full body
│   ├── faiss_index_v2/      # cached dense index (auto-generated)
│   ├── bm25_index_v2/       # cached sparse index (auto-generated)
│   ├── hyde_v2_docs/        # generated hypothetical headers
│   ├── hyde_v2_embeds/      # embeddings + averaged HyDE vectors
│   └── vectors/             # exported vectors + PCA/t-SNE visualization
├── src/                     # pipeline: embedder, hyde, retriever, evaluator
├── experiments_v2/          # RQ1 / RQ2 experiments (full corpus)
├── experiments_rq3/         # RQ3 experiments (curto / medio / longo)
├── results_v2/              # RQ1 / RQ2 metric outputs
├── results_rq3/             # RQ3 metric outputs
└── results_rq3_qwen/        # RQ3 metric outputs with Qwen 2.5 14b
```

---

## Running the Experiments

### RQ1 / RQ2 — main experiments (full corpus)

```bash
python experiments_v2/baseline_dense_v2.py
python experiments_v2/hyde_dense_v2.py
python experiments_v2/baseline_hybrid_v2.py
python experiments_v2/hyde_hybrid_v2.py
```

### RQ3 — query-granularity analysis

#### 1. Prepare queries by granularity

Extracts the three granularity levels from each document's structured body:

```bash
python -m src.prepare_rq3
```

Generates `queries_rq3_curto.json` (section I only), `queries_rq3_medio.json` (sections I + II), and `queries_rq3_longo.json` (full body).

#### 2. Generate hypothetical documents (HyDE) per granularity

```bash
python -m src.hyde_rq3 --granul curto
python -m src.hyde_rq3 --granul medio
python -m src.hyde_rq3 --granul longo
```

#### 3. Compute embeddings and averaged vectors

```bash
python -m src.hyde_embedder_rq3 --granul curto
python -m src.hyde_embedder_rq3 --granul medio
python -m src.hyde_embedder_rq3 --granul longo
```

#### 4. Run the retrieval experiments

FAISS and BM25 indexes are built automatically on first run. Use `&&` to chain runs:

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

- The `longo` scenario uses the complete structured body and is equivalent to the main RQ1 / RQ2 experiments — those results can be reused.
- Indexes are cached on disk. To force a rebuild (e.g. after changing the corpus), delete the relevant folders under `data/` (`faiss_index_v2/`, `bm25_index_v2/`) and the `hyde_*` caches.
- Baseline configurations do not invoke the LLM — only HyDE configurations do.

---

## Vector Export & Visualization

The intermediate embeddings can be exported for inspection and analysis:

```bash
python -m src.export_vectors --compute-queries
```

This produces raw CSVs (501 × 768) for each vector type — `f(ementa)`, `f(descrição)`, `f(mean HyDE)` — plus a 2D PCA scatter plot (`pca_2d.png`) showing how the three representations occupy the embedding space.

---

## Contributions

- A controlled benchmark for **dense and hybrid retrieval over structured Brazilian legal texts**.
- An empirical evaluation of **HyDE in Portuguese-language legal documents**.
- A study of how **query granularity** affects generation-based semantic expansion.
- A reproducible experimental pipeline for **legal information retrieval**.

---

## Research Context

This repository is part of an **undergraduate research project in Computer Engineering**, focusing on Information Retrieval and Natural Language Processing applied to the legal domain. The goal is to investigate modern retrieval techniques on Brazilian legal data and contribute reproducible benchmarks to the area.

---

## License

This repository uses only **publicly available legal documents**. Refer to the respective court systems for the original sources.