"""Завантаження outputs/generations.json з опційними фільтрами (див. conftest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "generations.json"

# Встановлює tests/conftest.py під час collection при -m lang_* / severity_*.
lang_filter: str | None = None
severity_filter: str | None = None


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip(
            "Спершу згенеруй outputs/generations.json: `python src/generate.py`"
        )
    records = json.loads(GENERATIONS.read_text(encoding="utf-8"))

    if lang_filter is not None:
        records = [r for r in records if r.get("lang") == lang_filter]
    if severity_filter is not None:
        records = [r for r in records if r.get("severity") == severity_filter]

    if not records and (lang_filter is not None or severity_filter is not None):
        parts = []
        if lang_filter is not None:
            parts.append(f"lang={lang_filter!r}")
        if severity_filter is not None:
            parts.append(f"severity={severity_filter!r}")
        pytest.skip(
            "Немає записів із "
            + ", ".join(parts)
            + " у generations.json"
        )
    return records
