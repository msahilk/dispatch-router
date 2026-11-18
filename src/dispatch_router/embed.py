"""Sentence embeddings for every prompt, cached to data/embeddings/.

We embed the full prompt exactly as it is sent to the model, since that is all a
router would see at request time. Vectors are L2-normalized, so a dot product is
cosine similarity.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import load_config, path


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(load_config()["embeddings"]["model"], device=device())
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True).astype(
        np.float32
    )


def save(name: str, df: pd.DataFrame, X: np.ndarray) -> None:
    d = path("embeddings_dir")
    np.save(d / f"{name}.npy", X)
    (d / f"{name}.ids.json").write_text(json.dumps(list(df["id"])))


def load(name: str, df: pd.DataFrame) -> np.ndarray:
    """Load cached embeddings, re-ordered to match df's ids."""
    d = path("embeddings_dir")
    X = np.load(d / f"{name}.npy")
    ids = json.loads((d / f"{name}.ids.json").read_text())
    pos = {i: k for k, i in enumerate(ids)}
    return X[[pos[i] for i in df["id"]]]
