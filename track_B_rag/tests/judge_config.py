"""
Конфігурація LLM-судді для Ragas.

Завантаження локального LLM-судді (Qwen3-8B / Qwen3-14B залежно від GPU_PROFILE;
4-bit GPU / fp16 MPS / fp32 CPU),
обгортки для Ragas LangchainLLMWrapper та ембедингів.

Модуль не залежить від pytest — піднімає стандартні винятки.
Тести самі вирішують, чи skip, чи fail.
"""

from __future__ import annotations

from typing import Any

from config import EMBED_MODEL_ID, GPU_PROFILE, JUDGE_MODEL_ID


class JudgeSetupError(RuntimeError):
    """Неможливо підняти LLM-суддю (відсутні залежності, модель, VRAM тощо)."""


class Qwen3NoThinkPipeline:
    """
    Ragas/LangChain подають звичайний str-промпт.
    Обгортаємо його в chat template з enable_thinking=False,
    щоб Ragas отримував короткий JSON/verdict, а не <think>…
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


def load_qwen3_judge() -> Qwen3NoThinkPipeline:
    """
    Локальний Qwen3 LLM-суддя для Ragas (модель з config.JUDGE_MODEL_ID).

    1xT4: Qwen3-8B; 2xT4: Qwen3-14B (dense — bnb 4-bit працює; MoE 30B — ні).
    4-bit через BitsAndBytesConfig на CUDA; без bnb — MPS/CPU fp16/bf16.
    Потрібен transformers>=4.51 (підтримка model_type=qwen3).
    max_new_tokens=1024, enable_thinking=False.

    Raises:
        JudgeSetupError: залежності відсутні або модель не завантажилась.
    """
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as ex:
        raise JudgeSetupError(f"transformers/torch недоступні: {ex!r}") from ex

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if ver < (4, 51):
        raise JudgeSetupError(
            f"transformers={transformers.__version__} не знає qwen3; "
            f"онови: pip install -U 'transformers>=4.51'"
        )

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}

    if GPU_PROFILE == "2xT4":
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            raise JudgeSetupError(
                f"GPU_PROFILE=2xT4, але torch бачить {n_gpus} GPU. "
                f"Перевір CUDA_VISIBLE_DEVICES або середовище."
            )

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            raise RuntimeError("bitsandbytes 4-bit потребує CUDA")
    except Exception:
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
        raise JudgeSetupError(
            f"не вдалося завантажити {JUDGE_MODEL_ID}: {type(ex).__name__}: {ex}"
        ) from ex

    _dtype = getattr(model, "dtype", "?")
    _dev = getattr(model, "device", next(model.parameters()).device)
    print(
        f"[judge_config] GPU_PROFILE={GPU_PROFILE}  model={JUDGE_MODEL_ID}  "
        f"model.dtype={_dtype}  device={_dev}  cuda_count={torch.cuda.device_count()}"
    )

    _orig_apply = tokenizer.apply_chat_template

    def _apply_chat_template_no_think(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return _orig_apply(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_chat_template_no_think

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0,
        return_full_text=False,
    )
    return Qwen3NoThinkPipeline(gen_pipeline)


def wrap_judge_for_ragas(gen_pipeline: Qwen3NoThinkPipeline):
    """LangChain-сумісний LLM → Ragas LangchainLLMWrapper.

    Raises:
        JudgeSetupError: langchain/ragas wrappers недоступні.
    """
    try:
        from langchain_huggingface import HuggingFacePipeline
        from ragas.llms import LangchainLLMWrapper
    except ImportError as ex:
        raise JudgeSetupError(f"langchain/ragas wrappers недоступні: {ex!r}") from ex

    lc_llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return LangchainLLMWrapper(lc_llm)


def wrap_embeddings_for_ragas():
    """Локальні e5-ембединги для AnswerRelevancy (без OpenAI).

    Raises:
        JudgeSetupError: embeddings wrappers недоступні.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as ex:
        raise JudgeSetupError(f"embeddings wrappers недоступні: {ex!r}") from ex

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb)


def build_ragas_metrics(ragas_llm, ragas_embeddings) -> list:
    """Faithfulness, Answer Relevancy/Correctness, Context Precision/Recall.

    Raises:
        JudgeSetupError: не вдалося зібрати Ragas-метрики.
    """
    try:
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
        raise JudgeSetupError(f"не вдалося зібрати Ragas-метрики: {ex!r}") from ex
