from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    # Reuse one configured client across all LLM calls in the process.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)
