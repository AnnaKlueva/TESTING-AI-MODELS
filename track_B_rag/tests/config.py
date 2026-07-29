"""
Централізована конфігурація eval-пайплайну.

Пороги метрик, ідентифікатори моделей, шляхи до артефактів.
Жодних важких залежностей (torch, ragas, …) — лише stdlib.
Єдине місце для тюнінгу; обґрунтуй у test_strategy.md (розділ 5 / 6).

GPU-профіль (env var GPU_PROFILE)
──────────────────────────────────
Визначає, як завантажується Qwen3-8B LLM-суддя і скільки Ragas-воркерів працюють.

  ┌──────────┬───────────┬──────────────────┬─────────────┐
  │ Профіль  │ dtype     │ квантизація      │ max_workers │
  ├──────────┼───────────┼──────────────────┼─────────────┤
  │ 1xT4     │ 4-bit bnb │ load_in_4bit=True│ 1           │
  │ 2xT4     │ float16   │ —                │ 2           │
  └──────────┴───────────┴──────────────────┴─────────────┘

Використання:

  # одна T4 (default)
  GPU_PROFILE=1xT4 python -m pytest tests/test_eval.py::test_ragas_qwen3_judge -v

  # дві T4
  GPU_PROFILE=2xT4 python -m pytest tests/test_eval.py::test_ragas_qwen3_judge -v

  # Colab notebook
  import os; os.environ["GPU_PROFILE"] = "2xT4"
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── GPU-профіль ─────────────────────────────────────────────────────────
#   GPU_PROFILE=1xT4  — одна T4 (16 GB), 4-bit квантизація, 1 воркер     (default)
#   GPU_PROFILE=2xT4  — дві T4 (2×16 GB), fp16 через обидві GPU, >1 воркер

GPU_PROFILE: str = os.environ.get("GPU_PROFILE", "1xT4").strip()

_VALID_GPU_PROFILES = {"1xT4", "2xT4"}
if GPU_PROFILE not in _VALID_GPU_PROFILES:
    raise ValueError(
        f"GPU_PROFILE={GPU_PROFILE!r} не підтримується. "
        f"Допустимі: {', '.join(sorted(_VALID_GPU_PROFILES))}"
    )

# ── Моделі ──────────────────────────────────────────────────────────────

JUDGE_MODEL_ID = "Qwen/Qwen3-8B"
EMBED_MODEL_ID = "intfloat/multilingual-e5-base"

# ── Шляхи до артефактів ─────────────────────────────────────────────────

RAGAS_RESULTS_JSON = ROOT / "outputs" / "rag_evaluation_results.json"
RAGAS_METRICS_LOG = ROOT / "outputs" / "ragas_metrics.log"
RETRIEVAL_METRICS_LOG = ROOT / "outputs" / "retrieval_metrics.log"

# ── Пороги: retrieval ───────────────────────────────────────────────────

PASS_RATE_THRESHOLD = 0.8

# ── Пороги: Ragas LLM-as-judge ──────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7
ANSWER_CORRECTNESS_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.7
CONTEXT_RECALL_THRESHOLD = 0.7
