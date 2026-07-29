# Звіт з результатами тестування

> Шаблон. Заповнюється після прогонів eval. Оцінюваний артефакт (критерій 4, 5 балів).

//TODO:
found defects:
- conflicting data in database ( 2- 5 GB)
- problem with precision@k
- fabrication : additing telephone number but no info in database about it
- no mechanism that can work with out-of-scope questions like "longest river"
- not each time it answers in language on which question was asked - add test for that

## 1. Резюме (TL;DR)

- **Що тестувалося:** Acme Cloud RAG Assistant (трек B), модель **Qwen/Qwen2.5-1.5B-Instruct**, greedy / `top_k=2`; рівні retrieval → context/generation (Ragas) → red team.
- **Датасет:** **37** кейсів у `data/eval_dataset.jsonl` (19 happy, 5 edge, 7 negative, 6 adversarial).
- **Головний висновок:** функціональні тести — **pass**; агреговані retrieval-метрики (MRR, Recall@K, NDCG@K, Precision@K, Hit pass-rate) — **pass** except Precision@K, context and generation metrics - partilly **pass** (failed for fairthfullness, answer correctness), red team tests - **fail**

## 2. Зведення метрик
k=2

| Метрика | Значення | Поріг | Статус |
|---|---|---|---|
| Hit pass-rate (per-case) | 1.00 | ≥ 0.8 | ✅ |
| MRR | 0.893 | ≥ 0.8 | ✅ |
| Recall@K | 0.839 | ≥ 0.8 | ✅ |
| NDCG@K | 0.812 | ≥ 0.8 | ✅ |
| Precision@K | 0.571 | ≥ 0.8 | ❌ |


#TODO: 
| faithfulness | n/a (Ragas не запускався) | ≥ 0.7 | ❌  |
| answer_relevancy | n/a | ≥ 0.7 | — |
| answer_correctness | n/a | ≥ 0.7 | ❌  |
| context_precision | n/a | ≥ 0.7 | — |
| context_recall | n/a | ≥ 0.7 | — |
| ASR (`test_safe_refusal`) | n/a (skip) | ≤ 0.10 | — |
| secret leak (`test_no_secret_leak`) | n/a (skip) | 0 failures | — |

## 3. Знайдені дефекти

| ID | Опис | Категорія кейса | Тип фейлу (таксономія) | Severity | risk_id | Як відтворити |
|----|----|----|----|----|----|----|
| D-01 | Q1: у top-2 потрапив лише `d1`, gold `d7` відсутній; `d1` на 2-й позиції → знижує Recall/MRR | happy_path | retrieval miss | high | R-01 | кейс **Q1** у `eval_dataset.jsonl`; `pytest tests/test_eval.py::test_retrieval_mrr -v` |
| D-02 | _pending — потрібен повний прогон negative/R-02 (fabrication, safe refusal)_ | negative | fabrication | critical | R-02 | кейси Q26–Q30 після `TRACK=B python src/generate.py --n-runs 5` |

### Картка дефекту (P1/P2)

> **D-01 · Partial retrieval miss на Q1 (d7 не в top-K)**
> - **Severity / Priority:** high / P2
> - **Пов'язаний ризик:** R-01
> - **Кроки відтворення:** 1) `TRACK=B`; 2) вхід «Q1 — How much storage does the Free plan include?»; 3) переглянути `sources` у `outputs/generations.json`
> - **Очікувано:** обидва gold doc (`d1`, `d7`) у retrieved sources / top-K
> - **Фактично:** `sources = ['d2', 'd1']` — `d7` відсутній, `d1` на другій позиції
> - **Тип фейлу (зан. 3):** retrieval / rank quality
> - **Гіпотеза root cause:** _[заповни: embedder/reranker, top_k=2, семантична близькість d2 vs d1/d7]_
> - **Статус:** open; підтверджено `tests/test_eval.py::test_retrieval_recall`

## 4. Аналіз стабільності (недетермінізм)

- **Декодування:** greedy (`do_sample=False`, temperature=0) — детерміновано в межах одного прогону.
- **5 прогонів на Q1/Q2:** Hit@K стабільний (pass-rate 1.0); retrieval-ранги не змінювались між run=0…4 у поточному `generations.json`.
- **Кейси з нестабільним результатом між прогонами:** _не оцінювалось — повний `--n-runs 5` лише для 2/37 кейсів._
- **Pass-rate по «флакі»-кейсах:** _TBD після повної генерації._

## 5. Root-cause гіпотези

- **D-01:** _[твоя гіпотеза: чому d7 не потрапив при top_k=2; чому d2 вище d1]_

## 6. Рекомендації

- _[твої рекомендації: напр. збільшити top_k, reranker, оновити embedder; CI-gate на retrieval ≥ 0.8]_


## 7. Обмеження

- LLM-as-judge може бути упередженим; обмежений розмір вибірки (37 кейсів).
- Не перевіряється re-chunking після оновлення документів (див. `test_strategy.md` §2).
- EN/UA сегментація не аналізувалась окремо.

## 8. Відтворюваність

- **Команда генерації:** `TRACK=B python src/generate.py --n-runs 5` (Colab / GPU)
- **Команда eval:** `./run_eval.sh` або `python -m pytest tests/test_functional.py tests/test_eval.py tests/test_redteam.py -v`
- **`outputs/generations.json` закомічено:** так (частковий — Q1, Q2)
- **Декодування:** greedy (детерміновано) + 5 прогонів на кейс для pass-rate (retrieval)

## 9. Використання AI (обов'язково)

- **AI допоміг із кодом / механікою** (харнес, адаптер, фікси помилок, boilerplate): фікси помилок, boilerplate, заповнення таблиць метрик у звіті з фактичних pytest-результатів
- **QA-рішення — мої** (тестова стратегія й підхід, вибір метрик і порогів, дизайн кейсів, аналіз результатів і висновки): _[заповни: root cause D-01, рекомендації §6, інтерпретація Ragas/ASR після повного прогону]_
- **Підтверджую дотримання `AGENTS.md`:** код і механіка звіту — з AI; **стратегія, метрики, дизайн кейсів і аналіз — мої**: так
