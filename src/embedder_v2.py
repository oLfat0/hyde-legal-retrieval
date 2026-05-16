"""
embedder_v2.py
--------------
Versao v2 do embedder — indexa o campo "ementa" (palavras-chave) no FAISS.

Diferenca principal em relacao ao embedder.py original:
  - Corpus lido de CORPUS_V2_PATH
  - Texto indexado = doc["ementa"]  (nao doc["texto"])
  - Indice salvo em FAISS_V2_DIR
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

from src.assets_v2 import CORPUS_V2_PATH, FAISS_V2_DIR

EMBEDDING_MODEL  = "facebook/mcontriever-msmarco"
FAISS_INDEX_PATH = Path(FAISS_V2_DIR) / "corpus.index"
CORPUS_META_PATH = Path(FAISS_V2_DIR) / "corpus_meta.pkl"
BATCH_SIZE       = 32
MAX_SEQ_LENGTH   = 512
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[AutoModel, AutoTokenizer]:
    print(f"[embedder_v2] Carregando {EMBEDDING_MODEL} em {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)
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
    all_embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch   = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(DEVICE)
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
    Constroi indice FAISS indexando o campo 'ementa' de cada documento.
    Retorna (index, cdacordao_list).
    """
    Path(FAISS_V2_DIR).mkdir(parents=True, exist_ok=True)

    if not force_rebuild and FAISS_INDEX_PATH.exists() and CORPUS_META_PATH.exists():
        print("[embedder_v2] Indice v2 encontrado em disco — carregando...")
        return load_index()

    print("[embedder_v2] Construindo indice v2 (campo: ementa)...")

    with open(CORPUS_V2_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # INDEXA APENAS A EMENTA (palavras-chave), nao a descricao
    texts          = [doc["ementa"]    for doc in corpus]
    cdacordao_list = [doc["cdacordao"] for doc in corpus]

    embeddings = encode_texts(texts, model, tokenizer, desc="Indexando ementas v2")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CORPUS_META_PATH, "wb") as f:
        pickle.dump(cdacordao_list, f)

    print(f"[embedder_v2] Indice salvo — {index.ntotal} ementas, dim={dim}")
    return index, cdacordao_list


def load_index() -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CORPUS_META_PATH, "rb") as f:
        cdacordao_list = pickle.load(f)
    print(f"[embedder_v2] Indice carregado — {index.ntotal} docs")
    return index, cdacordao_list


def dense_search(
    query_embeddings: np.ndarray,
    index: faiss.Index,
    cdacordao_list: list[str],
    top_k: int = 10,
) -> list[list[tuple[str, float]]]:
    scores_mat, indices_mat = index.search(query_embeddings, top_k)
    rankings = []
    for scores_row, indices_row in zip(scores_mat, indices_mat):
        ranking = [
            (cdacordao_list[idx], float(score))
            for idx, score in zip(indices_row, scores_row)
            if idx != -1
        ]
        rankings.append(ranking)
    return rankings