"""
Метричне оцінювання НАД збереженими генераціями (офлайн, детерміновано, без ключів).

Патерн з ДЗ 8: читаємо outputs/generations.json і рахуємо метрики БЕЗ виклику моделі.
- retrieval-метрики (трек B) рахуються чисто з gold_doc_ids — модель не потрібна взагалі;
- DeepEval використовуй через BaseMetric (детерміновано, без LLM-судді);
- Ragas — лише non-LLM і ТІЛЬКИ через локальний try/except (на Colab крихкий).

Недетермінізм: якщо генерував кілька прогонів (--n-runs), оцінюй PASS-RATE по run, а не один прогін.

⚠️ Скаффолд. Вибір метрик і порогів — твоє оцінюване рішення (RUBRIC критерій 3).
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATIONS = ROOT / "outputs" / "generations.json"

PASS_RATE_THRESHOLD = 0.8   # TODO: обґрунтуй у test_strategy.md (розділ 5)


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip("Спершу згенеруй outputs/generations.json: `python src/generate.py`")
    return json.loads(GENERATIONS.read_text(encoding="utf-8"))


def pass_rate_by_case(records, predicate) -> dict:
    """Частка прогонів, що проходять `predicate`, окремо для кожного case id (для стабільності)."""
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec["id"]].append(bool(predicate(rec)))
    return {cid: sum(v) / len(v) for cid, v in buckets.items()}


@pytest.mark.skip(reason="TODO: реалізуй метрики під свій трек")
def test_metric_meets_threshold():
    """
    Приклад каркаса: визнач predicate(rec) -> bool під обрану метрику,
    усередни по прогонах і порівняй з порогом.

    Трек B (RAG) — приклад retrieval БЕЗ моделі:
        def hit(rec): return any(d in rec.get('sources', []) for d in rec['gold_doc_ids'])
        rates = pass_rate_by_case(load_generations(), hit)
        assert min(rates.values()) >= PASS_RATE_THRESHOLD

    Трек A/C — визнач власні predicate (safe-refusal, exact-match по зрізах тощо).

    Трек D (agent) — tool call accuracy + trajectory БЕЗ моделі (з траси у generations.json):
        def tools_ok(rec):
            called = [t['name'] for t in rec.get('tool_calls', [])]
            return set(called) >= set(rec['expected_tools']) and rec['selected_agent'] == rec['expected_agent']
        # Траєкторію (петлі, надлишкові виклики) перевір ВЛАСНИМ предикатом над rec['tool_calls'] —
        # що саме вважати «надлишковим», вирішуєш ти (RUBRIC критерій 3).
        # rates = pass_rate_by_case(load_generations(), tools_ok); assert ...
    """
    ...


# Приклад інтеграції Ragas БЕЗ ключа і БЕЗ падіння на Colab (патерн з ДЗ 8):
def _maybe_ragas_context_metric():
    try:
        import ragas  # noqa: F401  (імпорт лише тут, не на верхньому рівні)
    except Exception:
        pytest.skip("ragas недоступний у цьому середовищі — пропускаємо (це нормально)")
    # TODO: non-LLM context-метрика, якщо потрібна
    ...
