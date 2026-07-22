"""
Метричне оцінювання НАД збереженими генераціями (офлайн, детерміновано, без ключів).

Патерн з ДЗ 8: читаємо outputs/generations.json і рахуємо метрики БЕЗ виклику моделі.
- retrieval-метрики рахуються чисто з gold_doc_ids;
- DeepEval — через BaseMetric (детерміновано, без LLM-судді);
- Ragas — лише non-LLM і ТІЛЬКИ через локальний try/except.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

GENERATIONS = ROOT / "outputs" / "generations.json"

# Пороги — обґрунтуй у test_strategy.md (розділ 5 / 6).
PASS_RATE_THRESHOLD = 0.8
FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.4
RAGAS_CONTEXT_FLOOR = 0.5

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


def load_generations() -> list[dict]:
    if not GENERATIONS.exists():
        pytest.skip("Спершу згенеруй outputs/generations.json: `python src/generate.py`")
    return json.loads(GENERATIONS.read_text(encoding="utf-8"))


def pass_rate_by_case(records, predicate) -> dict:
    """Частка прогонів, що проходять `predicate`, окремо для кожного case id."""
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec["id"]].append(bool(predicate(rec)))
    return {cid: sum(v) / len(v) for cid, v in buckets.items()}


def _answerable_with_gold(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("gold_doc_ids")]


#RETRIEVAL tests
def test_retrieval_hit_pass_rate():
    """Hit@K: хоч один gold_doc_id у sources; pass-rate по кейсах ≥ порогу."""
    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    def hit(rec):
        gold = rec.get("gold_doc_ids") or []
        sources = rec.get("sources") or []
        return any(d in sources for d in gold)

    rates = pass_rate_by_case(records, hit)
    failed = {cid: rate for cid, rate in rates.items() if rate < PASS_RATE_THRESHOLD}
    assert not failed, (
        f"Hit pass-rate < {PASS_RATE_THRESHOLD} для кейсів: {failed}"
    )


def test_retrieval_mrr():
    """Середній MRR по кейсах із gold."""
    from metrics.custom_metrics import reciprocal_rank, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    # Один прогін на id (перший), щоб не дублювати при --n-runs.
    seen = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)

    ranked_sources, gold_ids_list = [], []
    for rec in seen.values():
        sources = rec.get("sources") or []
        gold = rec["gold_doc_ids"]
        ranked_sources.append(sources)
        gold_ids_list.append(gold)

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_mrr = aggregate(reciprocal_rank, ranked_sources, gold_ids_list)
    print(f"\nMean MRR (k={k}): {mean_mrr:.3f}, threshold={PASS_RATE_THRESHOLD}")
    assert mean_mrr >= PASS_RATE_THRESHOLD, f"mean MRR={mean_mrr:.3f}"


def test_retrieval_recall():
    """Середній Recall@K по кейсах із gold."""
    from metrics.custom_metrics import recall_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    # Один прогін на id (перший), щоб не дублювати при --n-runs.
    seen = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)

    ranked_sources, gold_ids_list = [], []
    for rec in seen.values():
        sources = rec.get("sources") or []
        gold = rec["gold_doc_ids"]
        ranked_sources.append(sources)
        gold_ids_list.append(gold)

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_recall = aggregate(recall_at_k, ranked_sources, gold_ids_list, k=k)
    print(f"\nMean Recall@K (k={k}): {mean_recall:.3f}, threshold={PASS_RATE_THRESHOLD}")
    assert mean_recall >= PASS_RATE_THRESHOLD, f"mean Recall@K={mean_recall:.3f}"


def test_retrieval_ndcg():
    """Середній NDCG@K по кейсах із gold."""
    from metrics.custom_metrics import ndcg_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    # Один прогін на id (перший), щоб не дублювати при --n-runs.
    seen = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)

    ranked_sources, gold_ids_list = [], []
    for rec in seen.values():
        sources = rec.get("sources") or []
        gold = rec["gold_doc_ids"]
        ranked_sources.append(sources)
        gold_ids_list.append(gold)

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_ndcg = aggregate(ndcg_at_k, ranked_sources, gold_ids_list, k=k)
    print(f"\nMean NDCG@K (k={k}): {mean_ndcg:.3f}, threshold={PASS_RATE_THRESHOLD}")
    assert mean_ndcg >= PASS_RATE_THRESHOLD, f"mean NDCG@K={mean_ndcg:.3f}"


def test_retrieval_precision():
    """Середня Precision@K по кейсах із gold."""
    from metrics.custom_metrics import precision_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    # Один прогін на id (перший), щоб не дублювати при --n-runs.
    seen = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)

    ranked_sources, gold_ids_list = [], []
    for rec in seen.values():
        sources = rec.get("sources") or []
        gold = rec["gold_doc_ids"]
        ranked_sources.append(sources)
        gold_ids_list.append(gold)

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_precision = aggregate(precision_at_k, ranked_sources, gold_ids_list, k=k)
    print(f"\nMean Precision@K (k={k}): {mean_precision:.3f}, threshold={PASS_RATE_THRESHOLD}")
    assert mean_precision >= PASS_RATE_THRESHOLD, f"mean Precision@K={mean_precision:.3f}"


#CONTEXT + GENERATION tests

def _make_embed_fn():
    """Локальний e5-ембедер (як у ДЗ 8); skip якщо недоступний."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as ex:
        pytest.skip(f"sentence-transformers недоступний: {ex!r}")

    try:
        model = SentenceTransformer("intfloat/multilingual-e5-base")
    except Exception as ex:
        pytest.skip(f"не вдалося завантажити e5: {ex!r}")

    def embed_fn(texts):
        texts = list(texts)
        # e5: query/passage префікси не обов'язкові для проксі-порівнянь у тестах
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed_fn


def test_deepeval_proxy_metrics():
    """DeepEval BaseMetric-проксі Faithfulness + AnswerRelevancy без LLM-судді."""
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase

    from metrics.custom_metrics import (
        AnswerRelevancyProxyMetric,
        FaithfulnessProxyMetric,
    )

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    # Один запис на id; потрібні contexts (тексти).
    seen = {}
    for rec in records:
        if "contexts" not in rec:
            pytest.skip(
                "У generations.json немає 'contexts' — перегенеруй: python src/generate.py"
            )
        seen.setdefault(rec["id"], rec)

    embed_fn = _make_embed_fn()

    def both_ok(rec):
        tc = LLMTestCase(
            input=rec["input"],
            actual_output=rec.get("output") or "",
            retrieval_context=rec.get("contexts") or [],
        )
        mf = FaithfulnessProxyMetric(embed_fn, threshold=FAITHFULNESS_THRESHOLD)
        mr = AnswerRelevancyProxyMetric(embed_fn, threshold=ANSWER_RELEVANCY_THRESHOLD)
        mf.measure(tc)
        mr.measure(tc)
        return bool(mf.is_successful() and mr.is_successful())

    rates = pass_rate_by_case(list(seen.values()), both_ok)
    failed = {cid: rate for cid, rate in rates.items() if rate < PASS_RATE_THRESHOLD}
    assert not failed, (
        f"DeepEval proxy pass-rate < {PASS_RATE_THRESHOLD} для кейсів: {failed}"
    )


def test_ragas_non_llm_context():
    """Ragas non-LLM ContextRecall / ContextPrecision (rapidfuzz), без ключа."""
    try:
        import asyncio

        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
        )
    except Exception as ex:
        pytest.skip(f"ragas недоступний у цьому середовищі — пропускаємо: {ex!r}")

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    seen = {}
    for rec in records:
        if "contexts" not in rec:
            pytest.skip(
                "У generations.json немає 'contexts' — перегенеруй: python src/generate.py"
            )
        seen.setdefault(rec["id"], rec)

    rec_m = NonLLMContextRecall()
    prec_m = NonLLMContextPrecisionWithReference()
    recalls, precisions = [], []

    try:
        for rec in seen.values():
            ref = rec.get("reference_contexts") or []
            if not ref:
                # фолбек: текст gold з корпусу
                from metrics.custom_metrics import corpus_text_by_id

                id2text = corpus_text_by_id()
                ref = [
                    id2text[g] for g in (rec.get("gold_doc_ids") or []) if g in id2text
                ]
            if not ref:
                continue
            sample = SingleTurnSample(
                user_input=rec["input"],
                retrieved_contexts=rec.get("contexts") or [],
                reference_contexts=ref,
            )
            recalls.append(asyncio.run(rec_m.single_turn_ascore(sample)))
            precisions.append(asyncio.run(prec_m.single_turn_ascore(sample)))
    except Exception as ex:
        pytest.skip(f"ragas non-LLM (сумісність версій / rapidfuzz): {ex!r}")

    if not recalls:
        pytest.skip("Немає кейсів із reference_contexts для Ragas")

    mean_rec = sum(recalls) / len(recalls)
    mean_prec = sum(precisions) / len(precisions)
    assert mean_rec >= RAGAS_CONTEXT_FLOOR, f"Ragas ContextRecall={mean_rec:.3f}"
    assert mean_prec >= RAGAS_CONTEXT_FLOOR, f"Ragas ContextPrecision={mean_prec:.3f}"
