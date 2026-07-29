"""
Спільні pytest-хуки для треку B.

Фільтри даних через маркери (-m), комбінуються через and:

  python -m pytest -m lang_en -v
  python -m pytest -m severity_critical -v
  python -m pytest -m "lang_en and severity_critical" -v
  python -m pytest -m "smoke and severity_critical" -v

Без lang_* / severity_* у -m — усі записи, як раніше.
"""

from __future__ import annotations

import re

import pytest

import generations_loader

_LANG_MARKERS = {
    "lang_en": "en",
    "lang_uk": "uk",
}

_SEVERITY_MARKERS = {
    "severity_critical": "critical",
    "severity_high": "high",
    "severity_medium": "medium",
    "severity_low": "low",
}


def _markexpr_has(markexpr: str, marker_name: str) -> bool:
    return bool(
        re.search(rf"(?<![\w.]){re.escape(marker_name)}(?![\w.])", markexpr)
    )


def _apply_data_marker(
    markexpr: str,
    items: list[pytest.Item],
    markers: dict[str, str],
) -> str | None:
    """Якщо -m містить один із маркерів — вішаємо його на всі тести й повертаємо значення."""
    for marker_name, value in markers.items():
        if _markexpr_has(markexpr, marker_name):
            mark = getattr(pytest.mark, marker_name)
            for item in items:
                item.add_marker(mark)
            return value
    return None


_SUITE_MARKERS = {
    "smoke": "fast PR-gate checks (functional + critical retrieval + secret leak)",
    "regression": "full offline eval suite on committed generations.json",
    "functional": "contract / schema checks on SUT output",
    "retrieval": "retrieval metrics (MRR, Recall@k) against gold_doc_ids",
    "redteam": "adversarial / safety checks",
    "llm_as_judge": "Ragas metrics with local LLM judge (GPU, slow)",
}


def pytest_configure(config: pytest.Config) -> None:
    if config.option.log_cli_level is None:
        config.option.log_cli = True
        config.option.log_cli_level = "INFO"
        config.option.log_cli_format = "%(levelname)s %(message)s"
    for name, description in _SUITE_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    for name, code in _LANG_MARKERS.items():
        config.addinivalue_line(
            "markers",
            f"{name}: filter generations.json to cases with lang={code}",
        )
    for name, level in _SEVERITY_MARKERS.items():
        config.addinivalue_line(
            "markers",
            f"{name}: filter generations.json to cases with severity={level}",
        )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    lang_* / severity_* — фільтри даних, не окремі сьюті.
    Маркери вішаються на всі тести, щоб pytest не відсіяв їх через -m.
    """
    markexpr = (getattr(config.option, "markexpr", None) or "").strip()
    generations_loader.lang_filter = _apply_data_marker(
        markexpr, items, _LANG_MARKERS
    )
    generations_loader.severity_filter = _apply_data_marker(
        markexpr, items, _SEVERITY_MARKERS
    )
