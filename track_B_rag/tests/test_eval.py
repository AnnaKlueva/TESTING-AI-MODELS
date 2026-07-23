"""
Метричне оцінювання НАД збереженими генераціями.

1) Retrieval-метрики — офлайн з gold_doc_ids (без моделі).
2) Ragas LLM-as-judge — локальний Qwen3-8B (4-bit на GPU) над
   input / contexts / output / expected з outputs/generations.json
   (faithfulness, answer_relevancy, answer_correctness,
   context_precision, context_recall).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

GENERATIONS = ROOT / "outputs" / "generations.json"
RAGAS_RESULTS_JSON = ROOT / "outputs" / "rag_evaluation_results.json"

JUDGE_MODEL_ID = "Qwen/Qwen3-8B"
EMBED_MODEL_ID = "intfloat/multilingual-e5-base"

# Пороги — обґрунтуй у test_strategy.md (розділ 5 / 6).
PASS_RATE_THRESHOLD = 0.8
FAITHFULNESS_THRESHOLD = 0.8
ANSWER_RELEVANCY_THRESHOLD = 0.8
ANSWER_CORRECTNESS_THRESHOLD = 0.8
CONTEXT_PRECISION_THRESHOLD = 0.8
CONTEXT_RECALL_THRESHOLD = 0.8

# os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


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



def _one_run_per_id(records: list[dict]) -> list[dict]:
    """Перший прогін на id (щоб не дублювати при --n-runs)."""
    seen: dict[str, dict] = {}
    for rec in records:
        seen.setdefault(rec["id"], rec)
    return list(seen.values())


# ────────────────────── RETRIEVAL tests ─────────────────────────────────


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

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

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

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

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

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

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

    seen = _one_run_per_id(records)
    ranked_sources, gold_ids_list = [], []
    for rec in seen:
        ranked_sources.append(rec.get("sources") or [])
        gold_ids_list.append(rec["gold_doc_ids"])

    k = max(len(ranked_sources[0]) if ranked_sources else 0, 1)
    mean_precision = aggregate(precision_at_k, ranked_sources, gold_ids_list, k=k)
    print(f"\nMean Precision@K (k={k}): {mean_precision:.3f}, threshold={PASS_RATE_THRESHOLD}")
    assert mean_precision >= PASS_RATE_THRESHOLD, f"mean Precision@K={mean_precision:.3f}"

# CONTEXT + GENERATION tests
# ────────────────────── RAGAS + local Qwen3-8B judge ────────────────────


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
    except Exception as ex:
        pytest.skip(f"datasets недоступний: {ex!r}")

    rows = {
        # нові імена колонок (ragas ≥0.2)
        "user_input": [r["input"] for r in records],
        "retrieved_contexts": [list(r.get("contexts") or []) for r in records],
        "response": [r.get("output") or "" for r in records],
        "reference": [r.get("expected") or "" for r in records],
        # legacy-імена (ragas 0.1.x)
        "question": [r["input"] for r in records],
        "contexts": [list(r.get("contexts") or []) for r in records],
        "answer": [r.get("output") or "" for r in records],
        "ground_truth": [r.get("expected") or "" for r in records],
        "id": [r["id"] for r in records],
    }
    return Dataset.from_dict(rows)


def _load_qwen3_judge():
    """
    Локальний Qwen3-8B для Ragas.
    4-bit через BitsAndBytesConfig (Colab GPU); без bnb — MPS/CPU fp16/bf16.
    Потрібен transformers>=4.51 (підтримка model_type=qwen3).
    max_new_tokens=1024, enable_thinking=False (інакше Ragas часто дістає null).
    """
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as ex:
        pytest.skip(f"transformers/torch недоступні: {ex!r}")

    # Qwen3 з'явився в transformers≈4.51
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if ver < (4, 51):
        pytest.skip(
            f"transformers={transformers.__version__} не знає qwen3; "
            f"онови: pip install -U 'transformers>=4.51'"
        )

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            raise RuntimeError("bitsandbytes 4-bit потребує CUDA")
    except Exception:
        # Mac (MPS) / CPU: без 4-bit
        if torch.backends.mps.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_ID, **model_kwargs)
    except Exception as ex:
        pytest.skip(f"не вдалося завантажити {JUDGE_MODEL_ID}: {type(ex).__name__}: {ex}")

    # Qwen3: thinking увімкнений за замовчуванням у chat template —
    # вимикаємо, щоб Ragas отримував короткий JSON/verdict, а не <think>…
    _orig_apply = tokenizer.apply_chat_template

    def _apply_chat_template_no_think(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig_apply(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_chat_template_no_think

    # temperature=0 / do_sample=False — детермінований суддя
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0,
        return_full_text=False,
    )
    return _Qwen3NoThinkPipeline(gen_pipeline)


class _Qwen3NoThinkPipeline:
    """
    Ragas/LangChain подають звичайний str-промпт.
    Обгортаємо його в chat template з enable_thinking=False.
    """

    def __init__(self, inner):
        self._inner = inner
        self.tokenizer = inner.tokenizer
        self.model = inner.model

    def __call__(self, text_inputs, **kwargs):
        single = isinstance(text_inputs, str)
        prompts = [text_inputs] if single else list(text_inputs)
        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
        return self._inner(formatted[0] if single else formatted, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _wrap_judge_for_ragas(gen_pipeline):
    """LangChain-сумісний LLM → Ragas LangchainLLMWrapper."""
    try:
        from langchain_huggingface import HuggingFacePipeline
        from ragas.llms import LangchainLLMWrapper
    except Exception as ex:
        pytest.skip(f"langchain/ragas wrappers недоступні: {ex!r}")

    lc_llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return LangchainLLMWrapper(lc_llm)


def _wrap_embeddings_for_ragas():
    """Локальні e5-ембединги для AnswerRelevancy (без OpenAI)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as ex:
        pytest.skip(f"embeddings wrappers недоступні: {ex!r}")

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


def _build_ragas_metrics(ragas_llm, ragas_embeddings):
    """Faithfulness, Answer Relevancy/Correctness, Context Precision/Recall."""
    try:
        # ragas ≥0.2 class API
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        return [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ]
    except Exception:
        pass

    try:
        # ragas 0.1.x module-level metrics
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings
        answer_correctness.llm = ragas_llm
        answer_correctness.embeddings = ragas_embeddings
        context_precision.llm = ragas_llm
        context_recall.llm = ragas_llm
        return [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
        ]
    except Exception as ex:
        pytest.skip(f"не вдалося зібрати Ragas-метрики: {ex!r}")


def _export_ragas_results(result, records: list[dict]):
    """Результати → pandas DataFrame → JSON. Повертає DataFrame."""
    try:
        import pandas as pd
    except Exception as ex:
        pytest.skip(f"pandas недоступний: {ex!r}")

    # EvaluationResult у різних версіях ragas
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
    elif isinstance(result, dict):
        df = pd.DataFrame([result])
    else:
        try:
            df = pd.DataFrame(dict(result))
        except Exception:
            df = pd.DataFrame([dict(result)])

    # Підклеюємо id кейсів, якщо довжина збігається
    if len(df) == len(records) and "id" not in df.columns:
        df.insert(0, "id", [r["id"] for r in records])

    RAGAS_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(RAGAS_RESULTS_JSON, orient="records", force_ascii=False, indent=2)
    print(f"\nRagas results → {RAGAS_RESULTS_JSON}")
    print(df.to_string(index=False))
    return df


def _mean_ragas_metric(rows: list[dict], key: str) -> float | None:
    """Середнє по ключу; ігнорує None/NaN. None якщо немає валідних значень."""
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
    if not vals:
        return None
    return sum(vals) / len(vals)


def test_ragas_qwen3_judge():
    """
    Ragas LLM-as-judge: Faithfulness, Answer Relevancy, Answer Correctness,
    Context Precision, Context Recall з локальним Qwen3-8B.
    Зберігає per-example метрики у outputs/rag_evaluation_results.json.
    Assert: mean кожної метрики ≥ 0.8.

    Важкий тест (≈8B, GPU/Colab). Без моделі/VRAM — pytest.skip.
    Запуск окремо: pytest tests/test_eval.py::test_ragas_qwen3_judge -v -s
    """
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
    except Exception as ex:
        pytest.skip(f"ragas недоступний: {ex!r}")

    records = _records_for_ragas(load_generations())
    dataset = _to_hf_dataset(records)

    gen_pipeline = _load_qwen3_judge()
    ragas_llm = _wrap_judge_for_ragas(gen_pipeline)
    ragas_embeddings = _wrap_embeddings_for_ragas()
    metrics = _build_ragas_metrics(ragas_llm, ragas_embeddings)

    # Локальний Qwen3-8B: дефолт timeout=180s + max_workers=16 → TimeoutError.
    # Один воркер + довший timeout під послідовну HF-інференцію.
    run_config = RunConfig(timeout=600, max_workers=1, max_retries=3)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
    except TypeError:
        # старіші сигнатури evaluate()
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

    print("\n=== Ragas mean metrics (LLM judge) ===")
    failures: list[str] = []
    for key, threshold in checks:
        mean_val = _mean_ragas_metric(saved, key)
        if mean_val is None:
            print(f"{key}: n/a (усі значення null/NaN), threshold={threshold}")
            failures.append(f"{key}=n/a (немає валідних значень), threshold={threshold}")
            continue
        status = "PASS" if mean_val >= threshold else "FAIL"
        print(f"{key}: {mean_val:.4f} (threshold={threshold}) [{status}]")
        if mean_val < threshold:
            failures.append(f"{key}={mean_val:.4f} < {threshold}")

    assert not failures, "Ragas metrics below threshold:\n  - " + "\n  - ".join(failures)
