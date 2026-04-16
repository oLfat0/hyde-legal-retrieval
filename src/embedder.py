"""
embedder.py
-----------
Responsavel por:
  - Carregar mContriever (facebook/mcontriever-msmarco)
  - Computar embeddings de textos em batches
  - Construir e persistir indice FAISS (produto interno = cosseno após norm-L2)
  - Carregar indice ja existente (evita recomputacao entre experimentos)

O indice e salvo em data/faiss_index/ e reutilizado por TODOS os experimentos.
Nunca reconstrua o indice entre baseline e HyDE.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import os

from src.assets import CORPUS_PATH

# -- Configuracoes -------------------------------------------------------------
load_dotenv()

EMBEDDING_MODEL  = "facebook/mcontriever-msmarco"
FAISS_INDEX_DIR  = Path("data/faiss_index")
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "corpus.index"
CORPUS_META_PATH = FAISS_INDEX_DIR / "corpus_meta.pkl"
BATCH_SIZE       = 32
MAX_SEQ_LENGTH   = 512
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN         = os.getenv("HF_TOKEN")


def load_model() -> tuple[AutoModel, AutoTokenizer]:
    """Carrega mContriever-msmarco e tokenizador."""
    print(f"[embedder] Carregando {EMBEDDING_MODEL} em {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, token=HF_TOKEN)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, token=HF_TOKEN).to(DEVICE)
    model.eval()
    return model, tokenizer


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded  = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask       = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = BATCH_SIZE,
    desc: str = "Encoding",
) -> np.ndarray:
    """Retorna array float32 (N, D) com embeddings L2-normalizados."""
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch   = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        outputs    = model(**encoded)
        embeddings = _mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def build_index(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    force_rebuild: bool = False,
) -> tuple[faiss.Index, list[str]]:
    """
    Constroi o indice FAISS a partir do corpus de ementas.

    Retorna:
        index          : IndexFlatIP (produto interno = cosseno apos norm-L2)
        cdacordao_list : lista mapeando posicao no indice -> cdacordao

    Na primeira execucao: computa embeddings e salva em disco.
    Nas seguintes: carrega do disco (force_rebuild=False).
    """
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and FAISS_INDEX_PATH.exists() and CORPUS_META_PATH.exists():
        print("[embedder] Indice FAISS encontrado em disco - carregando...")
        return load_index()

    print("[embedder] Construindo indice FAISS a partir do corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    texts          = [doc["texto"]     for doc in corpus]
    cdacordao_list = [doc["cdacordao"] for doc in corpus]

    embeddings = encode_texts(texts, model, tokenizer, desc="Indexando corpus")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CORPUS_META_PATH, "wb") as f:
        pickle.dump(cdacordao_list, f)

    print(f"[embedder] Indice salvo - {index.ntotal} docs, dim={dim}")
    return index, cdacordao_list


def load_index() -> tuple[faiss.Index, list[str]]:
    """Carrega indice FAISS e mapeamento posicao -> cdacordao do disco."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CORPUS_META_PATH, "rb") as f:
        cdacordao_list = pickle.load(f)
    print(f"[embedder] Indice carregado - {index.ntotal} docs")
    return index, cdacordao_list


def dense_search(
    query_embeddings: np.ndarray,
    index: faiss.Index,
    cdacordao_list: list[str],
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Busca densa para N queries em batch.

    Args:
        query_embeddings : array (N, D) L2-normalizado
        index            : indice FAISS
        cdacordao_list   : mapeamento posicao -> cdacordao
        top_k            : candidatos por query

    Retorna:
        Lista de N rankings: [(cdacordao, score), ...] decrescente por score.
    """
    scores_mat, indices_mat = index.search(query_embeddings, top_k)

    rankings: list[list[tuple[str, float]]] = []
    for scores_row, indices_row in zip(scores_mat, indices_mat):
        ranking = [
            (cdacordao_list[idx], float(score))
            for idx, score in zip(indices_row, scores_row)
            if idx != -1
        ]
        rankings.append(ranking)

    return rankings