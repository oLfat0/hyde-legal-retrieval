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

Approximately **300 public legal decisions** extracted from the jurisprudence system of the Mato Grosso do Sul Court of Justice (TJMS).

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
├── results/            # evaluation results and metric outputs
└── paper/              # research manuscript
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

---

## License

This repository uses only **publicly available legal documents**. Please refer to the respective court systems for the original sources.