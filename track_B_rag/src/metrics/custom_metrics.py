"""
Кастомні метрики RAG : ретрівал-формули + семантичні проксі
+ DeepEval BaseMetric без LLM-судді й без API-ключів.
"""

from __future__ import annotations

import math
import re

try:
    from deepeval.metrics import BaseMetric
except Exception:  # deepeval optional for pure retrieval formulas
    BaseMetric = object  # type: ignore[misc, assignment]

# ────────────────────── РЕТРІВАЛ-МЕТРИКИ ─────────────────────────────────


def hit_rate_at_k(ranked_ids: list, gold_ids: list, k: int) -> float:
    """1.0, якщо хоч один gold у top-k (для одного запиту)."""
    gold = set(gold_ids)
    return 1.0 if any(r in gold for r in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: list, gold_ids: list, k: int) -> float:
    """Яка частка top-k результатів релевантна (мінімізує шум у контексті)."""
    if k <= 0:
        return 0.0
    gold = set(gold_ids)
    topk = ranked_ids[:k]
    return sum(1 for r in topk if r in gold) / k


def recall_at_k(ranked_ids: list, gold_ids: list, k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & gold) / len(gold)


def reciprocal_rank(ranked_ids, gold_ids):
    gold = set(gold_ids)
    for i, r in enumerate(ranked_ids, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0    


def mrr(list_of_ranked: list, list_of_gold: list) -> float:
    rrs = [reciprocal_rank(r, g) for r, g in zip(list_of_ranked, list_of_gold)]
    return sum(rrs) / len(rrs) if rrs else 0.0


def dcg_at_k(ranked_ids: list, gold_ids: list, k: int) -> float:
    gold = set(gold_ids)
    return sum(
        (1.0 if r in gold else 0.0) / math.log2(i + 1)
        for i, r in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(ranked_ids: list, gold_ids: list, k: int) -> float:
    dcg = dcg_at_k(ranked_ids, gold_ids, k)
    n_rel = min(len(set(gold_ids)), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0
    

def aggregate(metric_fn, ranked_list: list, gold_list: list, k: int | None = None) -> float:
    """Середнє metric_fn по всіх запитах """
    scores = []
    for ranked, gold in zip(ranked_list, gold_list):
        if k is None:
            scores.append(float(metric_fn(ranked, gold)))
        else:
            scores.append(float(metric_fn(ranked, gold, k)))
    return sum(scores) / len(scores) if scores else 0.0


# ────────────────────── СЕМАНТИЧНІ ПРОКСІ ────────────────────────────────


def cosine(a, b) -> float:
    da = math.sqrt(sum(x * x for x in a)) or 1e-9
    db = math.sqrt(sum(x * x for x in b)) or 1e-9
    return sum(x * y for x, y in zip(a, b)) / (da * db)


_SENT = re.compile(r"[^.!?…]+[.!?…]+|\S[^.!?…]*$")


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text or "") if s.strip()]


def faithfulness_semantic(answer, contexts, embed_fn, threshold: float = 0.6) -> float:
    """Проксі Faithfulness: частка тверджень, семантично підкріплених контекстом."""
    claims = split_claims(answer)
    if not claims:
        return 1.0
    ctx_vecs = embed_fn(contexts) if contexts else []
    if not ctx_vecs:
        return 0.0
    claim_vecs = embed_fn(claims)
    supported = 0
    for cv in claim_vecs:
        if max((cosine(cv, xv) for xv in ctx_vecs), default=0.0) >= threshold:
            supported += 1
    return supported / len(claims)


def answer_relevancy_semantic(question, answer, embed_fn) -> float:
    """Проксі Answer Relevancy: косинус питання↔відповідь."""
    qv, av = embed_fn([question, answer])
    return max(0.0, cosine(qv, av))


def corpus_text_by_id() -> dict[str, str]:
    """id → текст з корпусу SUT (для reference_contexts за потреби)."""
    from rag_sut import _CORPUS

    return {item["id"]: item["text"] for item in _CORPUS}