"""
Метричне оцінювання НАД збереженими генераціями.

1) Retrieval-метрики — офлайн з gold_doc_ids (без моделі).
2) Ragas LLM-as-judge — локальний Qwen3 (8B / 30B-Instruct за GPU_PROFILE) над
   input / contexts / output / expected з outputs/generations.json
   (faithfulness, answer_relevancy, answer_correctness,
   context_precision, context_recall).

Конфігурація: tests/config.py (пороги, шляхи, моделі).
LLM-суддя:   tests/judge_config.py (завантаження моделі, обгортки).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from config import (
    ANSWER_CORRECTNESS_THRESHOLD,
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXT_PRECISION_THRESHOLD,
    CONTEXT_RECALL_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    GPU_PROFILE,
    PASS_RATE_THRESHOLD,
    RAGAS_METRICS_LOG,
    RAGAS_RESULTS_JSON,
    RETRIEVAL_METRICS_LOG,
)

logger = logging.getLogger(__name__)
from generations_loader import load_generations
from judge_config import (
    JudgeSetupError,
    build_ragas_metrics,
    load_qwen3_judge,
    wrap_embeddings_for_ragas,
    wrap_judge_for_ragas,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


# ────────────────────── helpers ─────────────────────────────────────────

def pass_rate_by_case(records, predicate) -> dict:
    """Частка прогонів, що проходять `predicate`, окремо для кожного case id."""
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec["id"]].append(bool(predicate(rec)))
    return {cid: sum(v) / len(v) for cid, v in buckets.items()}


def _answerable_with_gold(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("gold_doc_ids")]


def _one_run_per_id(records: list[dict]) -> list[dict]:
    """Перший прогін на id (щоб не дублювати при --n-runs)."""
    seen: dict[str, dict] = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)
    return list(seen.values())


def _records_for_ragas(records: list[dict]) -> list[dict]:
    """Кейси з input / contexts / output; один прогін на id."""
    usable = []
    for rec in _one_run_per_id(records):
        if "contexts" not in rec:
            pytest.skip(
                "У generations.json немає 'contexts' — перегенеруй: python src/generate.py"
            )
        if not (rec.get("input") and rec.get("output") is not None):
            continue
        usable.append(rec)
    if not usable:
        pytest.skip("Немає записів із input/contexts/output для Ragas")
    return usable


def _to_hf_dataset(records: list[dict]):
    """Конвертує generations → Hugging Face Dataset (колонки Ragas)."""
    try:
        from datasets import Dataset
    except ImportError as ex:
        pytest.skip(f"datasets недоступний: {ex!r}")

    rows = {
        "user_input": [r["input"] for r in records],
        "retrieved_contexts": [list(r.get("contexts") or []) for r in records],
        "response": [r.get("output") or "" for r in records],
        "reference": [r.get("expected") or "" for r in records],
        "question": [r["input"] for r in records],
        "contexts": [list(r.get("contexts") or []) for r in records],
        "answer": [r.get("output") or "" for r in records],
        "ground_truth": [r.get("expected") or "" for r in records],
        "id": [r["id"] for r in records],
    }
    return Dataset.from_dict(rows)


def _export_ragas_results(result, records: list[dict]):
    """Результати → pandas DataFrame → JSON. Повертає DataFrame."""
    try:
        import pandas as pd
    except ImportError as ex:
        pytest.skip(f"pandas недоступний: {ex!r}")

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
    elif isinstance(result, dict):
        df = pd.DataFrame([result])
    else:
        try:
            df = pd.DataFrame(dict(result))
        except Exception:
            df = pd.DataFrame([dict(result)])

    if len(df) == len(records) and "id" not in df.columns:
        df.insert(0, "id", [r["id"] for r in records])

    RAGAS_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(RAGAS_RESULTS_JSON, orient="records", force_ascii=False, indent=2)
    return df


def _mean_ragas_metric(
    rows: list[dict], key: str
) -> tuple[float | None, int, int]:
    """
    Середнє по ключу; ігнорує None/NaN.
    Повертає (mean | None, n_scored, n_total).
    """
    n_total = len(rows)
    vals: list[float] = []
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        vals.append(fv)
    n_scored = len(vals)
    if not vals:
        return None, n_scored, n_total
    return sum(vals) / n_scored, n_scored, n_total


def _write_ragas_metrics_log(lines: list[str]) -> None:
    """Пише summary метрик у outputs/ragas_metrics.log."""
    RAGAS_METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    RAGAS_METRICS_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _dominant_script_lang(text: str) -> str | None:
    """
    Груба детекція en/uk за домінуючим алфавітом (офлайн, без langdetect).
    Повертає None, якщо в тексті немає літер для класифікації.
    """
    cyr = len(_CYRILLIC_RE.findall(text or ""))
    lat = len(_LATIN_RE.findall(text or ""))
    if cyr == 0 and lat == 0:
        return None
    if cyr > lat:
        return "uk"
    if lat > cyr:
        return "en"
    return None


def _question_lang(rec: dict) -> str | None:
    """Мова запиту: з input, інакше з метаданих кейсу (`lang`)."""
    detected = _dominant_script_lang(rec.get("input") or "")
    if detected is not None:
        return detected
    lang = rec.get("lang")
    return lang if lang in {"en", "uk"} else None


def _answer_matches_question_language(rec: dict) -> bool:
    """True, якщо output домінує тією ж мовою, що й запит."""
    q_lang = _question_lang(rec)
    a_lang = _dominant_script_lang(rec.get("output") or "")
    if q_lang is None or a_lang is None:
        return True
    return q_lang == a_lang


def _log_retrieval_metric(
    metric: str,
    value: float,
    *,
    k: int | None = None,
    threshold: float,
    extra: str = "",
) -> None:
    """Логує значення метрики завжди (pass/fail), у консоль і retrieval_metrics.log."""
    k_part = f" (k={k})" if k is not None else ""
    suffix = f", {extra.lstrip(', ')}" if extra else ""
    msg = f"mean {metric}={value:.3f}{k_part}, threshold={threshold}{suffix}"
    logger.info(msg)
    RETRIEVAL_METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RETRIEVAL_METRICS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


@pytest.fixture(scope="session", autouse=True)
def _retrieval_metrics_log_header() -> None:
    """Скидає retrieval_metrics.log на початку сесії pytest."""
    RETRIEVAL_METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    RETRIEVAL_METRICS_LOG.write_text(
        f"=== Retrieval metrics ({ts}) ===\n",
        encoding="utf-8",
    )


# ────────────────────── RETRIEVAL tests ─────────────────────────────────

@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.retrieval
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
    mean_hit = sum(rates.values()) / len(rates) if rates else 0.0
    _log_retrieval_metric(
        "Hit pass-rate",
        mean_hit,
        threshold=PASS_RATE_THRESHOLD,
        extra=f"cases={len(rates)}, failed_cases={failed or 'none'}",
    )
    assert not failed, (
        f"Hit pass-rate < {PASS_RATE_THRESHOLD} для кейсів: {failed}"
    )


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.retrieval
def test_retrieval_mrr():
    """Середній MRR по кейсах із gold."""
    from metrics.custom_metrics import reciprocal_rank, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_mrr = aggregate(reciprocal_rank, ranked_sources, gold_ids_list)
    _log_retrieval_metric("MRR", mean_mrr, k=k, threshold=PASS_RATE_THRESHOLD)
    assert mean_mrr >= PASS_RATE_THRESHOLD, f"mean MRR={mean_mrr:.3f}"


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.retrieval
def test_retrieval_recall():
    """Середній Recall@K по кейсах із gold."""
    from metrics.custom_metrics import recall_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_recall = aggregate(recall_at_k, ranked_sources, gold_ids_list, k=k)
    _log_retrieval_metric("Recall@K", mean_recall, k=k, threshold=PASS_RATE_THRESHOLD)
    assert mean_recall >= PASS_RATE_THRESHOLD, f"mean Recall@K={mean_recall:.3f}"


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.retrieval
def test_retrieval_ndcg():
    """Середній NDCG@K по кейсах із gold."""
    from metrics.custom_metrics import ndcg_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_ndcg = aggregate(ndcg_at_k, ranked_sources, gold_ids_list, k=k)
    _log_retrieval_metric("NDCG@K", mean_ndcg, k=k, threshold=PASS_RATE_THRESHOLD)
    assert mean_ndcg >= PASS_RATE_THRESHOLD, f"mean NDCG@K={mean_ndcg:.3f}"


@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.retrieval
@pytest.mark.xfail(
    reason="D-01: partial retrieval miss Q1 (d7 not in top-K), Precision@K=0.571",
    strict=False,
)
def test_retrieval_precision():
    """Середня Precision@K по кейсах із gold."""
    from metrics.custom_metrics import precision_at_k, aggregate

    records = _answerable_with_gold(load_generations())
    if not records:
        pytest.skip("Немає кейсів із gold_doc_ids")

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_precision = aggregate(precision_at_k, ranked_sources, gold_ids_list, k=k)
    _log_retrieval_metric(
        "Precision@K", mean_precision, k=k, threshold=PASS_RATE_THRESHOLD
    )
    assert mean_precision >= PASS_RATE_THRESHOLD, f"mean Precision@K={mean_precision:.3f}"


# ────────────────────── GENERATION tests ────────────────────────────────

@pytest.mark.smoke
@pytest.mark.regression
@pytest.mark.xfail(
    reason="D-05: відповідь не мовою запиту (Q2, Q16, Q17, Q27, Q29)",
    strict=False,
)
def test_answer_language_matches_question():
    """
    Відповідь має бути тією ж мовою, що й запит (en/uk).
    Оракул: домінуючий алфавіт у input vs output; tie-break — поле lang кейсу.
    Pass-rate по кейсах ≥ PASS_RATE_THRESHOLD (n-runs усереднюються через pass_rate_by_case).
    """
    records = [
        r for r in load_generations()
        if (r.get("input") or "").strip() and r.get("output") is not None
    ]
    if not records:
        pytest.skip("Немає записів із input/output у generations.json")

    rates = pass_rate_by_case(records, _answer_matches_question_language)
    failed = {cid: rate for cid, rate in rates.items() if rate < PASS_RATE_THRESHOLD}
    mean_rate = sum(rates.values()) / len(rates) if rates else 0.0
    _log_retrieval_metric(
        "Language match pass-rate",
        mean_rate,
        threshold=PASS_RATE_THRESHOLD,
        extra=f"cases={len(rates)}, failed_cases={failed or 'none'}",
    )
    assert not failed, (
        f"Language mismatch pass-rate < {PASS_RATE_THRESHOLD} для кейсів: {failed}"
    )


# ────────────────────── RAGAS LLM-as-judge ──────────────────────────────

@pytest.mark.regression
@pytest.mark.llm_as_judge
@pytest.mark.xfail(
    reason=(
        "D-02, D-06, D-08: answer_correctness / faithfulness below threshold "
        "(Q1, Q28, Q30, Q31)"
    ),
    strict=False,
)
def test_ragas_qwen3_judge():
    """
    Ragas LLM-as-judge: Faithfulness, Answer Relevancy, Answer Correctness,
    Context Precision, Context Recall з локальним Qwen3-суддею (див. config.JUDGE_MODEL_ID).
    Зберігає per-example метрики у outputs/rag_evaluation_results.json
    і summary (mean + n_scored/n_total) у outputs/ragas_metrics.log.
    Assert: mean кожної метрики ≥ порогу.

    Важкий тест (≈8B, GPU/Colab). Без моделі/VRAM — pytest.skip.
    Запуск окремо: pytest tests/test_eval.py::test_ragas_qwen3_judge -v
    """
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
    except ImportError as ex:
        pytest.skip(f"ragas недоступний: {ex!r}")

    records = _records_for_ragas(load_generations())
    dataset = _to_hf_dataset(records)

    try:
        gen_pipeline = load_qwen3_judge()
        ragas_llm = wrap_judge_for_ragas(gen_pipeline)
        ragas_embeddings = wrap_embeddings_for_ragas()
        metrics = build_ragas_metrics(ragas_llm, ragas_embeddings)
    except JudgeSetupError as ex:
        pytest.skip(str(ex))

    # 1xT4: Qwen3-8B 4-bit, один воркер.
    # 2xT4: Qwen3-30B-Instruct 4-bit через обидві GPU, 2 воркери паралельно.
    _max_workers = 2 if GPU_PROFILE == "2xT4" else 1
    run_config = RunConfig(timeout=600, max_workers=_max_workers, max_retries=3)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=True,
            run_config=run_config,
        )
    except TypeError:
        result = evaluate(dataset=dataset, metrics=metrics)
    except Exception as ex:
        pytest.skip(f"Ragas evaluate впав (GPU/пам'ять/сумісність): {ex!r}")

    _export_ragas_results(result, records)

    assert RAGAS_RESULTS_JSON.exists(), f"немає {RAGAS_RESULTS_JSON}"
    saved = json.loads(RAGAS_RESULTS_JSON.read_text(encoding="utf-8"))
    assert saved, "порожній rag_evaluation_results.json"

    checks = [
        ("faithfulness", FAITHFULNESS_THRESHOLD),
        ("answer_relevancy", ANSWER_RELEVANCY_THRESHOLD),
        ("answer_correctness", ANSWER_CORRECTNESS_THRESHOLD),
        ("context_precision", CONTEXT_PRECISION_THRESHOLD),
        ("context_recall", CONTEXT_RECALL_THRESHOLD),
    ]

    log_lines = [
        "=== Ragas mean metrics (LLM judge) ===",
        f"results_json={RAGAS_RESULTS_JSON}",
        f"n_cases={len(saved)}",
    ]
    failures: list[str] = []
    for key, threshold in checks:
        mean_val, n_scored, n_total = _mean_ragas_metric(saved, key)
        coverage = f"n_scored/n_total={n_scored}/{n_total}"
        if mean_val is None:
            line = (
                f"{key}: n/a (усі значення null/NaN), {coverage}, "
                f"threshold={threshold}"
            )
            log_lines.append(line)
            failures.append(
                f"{key}=n/a (немає валідних значень), {coverage}, threshold={threshold}"
            )
            continue
        status = "PASS" if mean_val >= threshold else "FAIL"
        line = (
            f"{key}: {mean_val:.4f} ({coverage}) "
            f"(threshold={threshold}) [{status}]"
        )
        log_lines.append(line)
        if mean_val < threshold:
            failures.append(
                f"{key}={mean_val:.4f} < {threshold} ({coverage})"
            )

    if failures:
        log_lines.append("failures:")
        log_lines.extend(f"  - {f}" for f in failures)
    else:
        log_lines.append("failures: none")

    _write_ragas_metrics_log(log_lines)

    assert not failures, "Ragas metrics below threshold:\n  - " + "\n  - ".join(failures)
