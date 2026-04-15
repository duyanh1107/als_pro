from __future__ import annotations

import json
import re
from pathlib import Path

from core.learner import Learner


BASE_DIR = Path(__file__).resolve().parent.parent
LEARNERS_DIR = BASE_DIR / "data" / "learners"


def normalize_simple_id(simple_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", simple_id.strip().lower())
    return normalized.strip("_")


def get_learner_path(simple_id: str) -> Path:
    LEARNERS_DIR.mkdir(parents=True, exist_ok=True)
    # The normalized simple ID becomes the stable filename for that learner profile.
    return LEARNERS_DIR / f"{normalize_simple_id(simple_id)}.json"


def load_profile(simple_id: str) -> dict | None:
    path = get_learner_path(simple_id)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_learner(simple_id: str, learner: Learner, name: str) -> None:
    path = get_learner_path(simple_id)
    payload = {
        "name": name,
        "simple_id": normalize_simple_id(simple_id),
        "learner": learner.to_dict(),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def create_new_learner(simple_id: str) -> Learner:
    # New learners start below medium mastery through the default BKT prior of 0.3.
    return Learner(normalize_simple_id(simple_id))
