"""Shared model loading with actionable errors for the UI and CLI."""

import httpx


def load_embedding_model(name):
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(name)
    except (httpx.HTTPError, OSError):
        raise RuntimeError(
            "Could not load the embedding model. Check Hugging Face download access. "
            "After downloading weights, HF_HUB_OFFLINE=1 permits loading from the local cache."
        ) from None


def load_reranking_model(name):
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(name, max_length=512)
    except (httpx.HTTPError, OSError):
        raise RuntimeError(
            "Could not load the reranking model. Check Hugging Face download access. "
            "After downloading weights, HF_HUB_OFFLINE=1 permits loading from the local cache."
        ) from None
