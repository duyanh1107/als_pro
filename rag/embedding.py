from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


# Load `.env` once at import time so command-line runs work without extra setup.
load_dotenv(override=True)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    # Cache the client so every embedding/query call reuses the same configuration.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    # Embeddings are the shared representation used for both chunks and user queries.
    response = get_openai_client().embeddings.create(model=model, input=text)
    return response.data[0].embedding
