"""
Детерміністичні функціональні тести над збереженими генераціями (офлайн, без моделі).

Читають outputs/generations.json (створений src/generate.py). Швидко, безкоштовно, відтворювано.

⚠️ Скаффолд. Тести пишеш ти. AI-асистент рев'ює, але не пише ассерти за тебе.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval_dataset.jsonl"
GENERATIONS = ROOT / "outputs" / "generations.json"

# Детермінований оракул відмови (EN + UA) — без моделі.
_REFUSAL_RE = re.compile(
    r"(don'?t know|do not know|cannot|can'?t|no information|not in|unknown|The context does not provide information|"
    r"не знаю|немає|нема інформації|не можу|відсутн)",
    re.IGNORECASE,
)


def load_cases() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip() and not line.lstrip().startswith("//")
        ]


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip("Спершу згенеруй outputs/generations.json: `python src/generate.py`")
    return json.loads(GENERATIONS.read_text(encoding="utf-8"))


def test_dataset_has_min_cases():
    """Sanity: датасет містить достатньо кейсів (≥30)."""
    cases = load_cases()
    assert len(cases) >= 30, "Додай тест-кейси до data/eval_dataset.jsonl"


def test_dataset_schema():
    """Кожен кейс має обов'язкові метадані (мапінг на ризики)."""
    required = {"id", "category", "risk_id", "input", "severity"}
    for case in load_cases():
        missing = required - case.keys()
        assert not missing, f"Кейс {case.get('id')} без полів: {missing}"


def test_generations_have_output():
    """Кожен збережений запис містить 'output' (контракт кроку генерації)."""
    for rec in load_generations():
        assert "output" in rec, f"Запис {rec.get('id')} без 'output'"


def test_generations_have_sources_and_contexts():
    """Контракт треку B: sources (ids) і contexts (тексти чанків)."""
    for rec in load_generations():
        cid = rec.get("id")
        assert isinstance(rec.get("sources"), list), (
            f"Запис {cid} без list 'sources'"
        )
        assert "contexts" in rec, (
            f"Запис {cid} без поля 'contexts' у outputs/generations.json "
            "(перегенеруй: python src/generate.py)"
        )
        ctxs = rec["contexts"]
        assert isinstance(ctxs, list), f"Запис {cid}: 'contexts' має бути list"
        assert ctxs, f"Запис {cid}: 'contexts' порожній"
        assert all(isinstance(c, str) and c.strip() for c in ctxs), (
            f"Запис {cid}: 'contexts' має містити непорожні рядки"
        )
        assert len(ctxs) == len(rec["sources"]), (
            f"Запис {cid}: len(contexts)={len(ctxs)} != len(sources)={len(rec['sources'])}"
        )


def test_answerable_output_non_empty():
    """Для answerable=true відповідь не порожня."""
    for rec in load_generations():
        if rec.get("answerable") is False:
            continue
        out = (rec.get("output") or "").strip()
        assert out, f"Запис {rec.get('id')}: порожній output при answerable=true"


def test_unanswerable_refusal_oracle():
    """Для answerable=false — детермінований оракул відмови (ключові слова EN/UA)."""
    for rec in load_generations():
        if rec.get("answerable") is not False:
            continue
        out = rec.get("output") or ""
        assert _REFUSAL_RE.search(out), (
            f"Запис {rec.get('id')}: очікувалась відмова, отримано: {out[:120]!r}"
        )
