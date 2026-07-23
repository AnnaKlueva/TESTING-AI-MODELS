# Документ тестової стратегії

> Шаблон. Заповни кожен розділ. Це оцінюваний артефакт (критерій 1, 5 балів).
> AI-асистент може пояснювати концепції, але **текст пишеш ти**.

## 1. Система під тестом (SUT)

- Обраний трек: _ B (rag)_
- Демо-система: Acme Cloud RAG Assistant
- **Зафіксована модель/версія:** Qwen/Qwen2.5-1.5B-Instruct
- Параметри генерації: top_k=2, temperature не застосовна / greedy (do_sample=False).

## 2. Цілі та межі тестування

- Що тестуємо: система RAG **на всіх рівнях** — retrieval → context → generation → end-to-end
- Що поза скоупом:   
- no verification for situation when doc gets re-chunked after an update  
- no comparison with previous test runs



## 3. Матриця ризиків (3–5 критичних точок)


| ID   | Точка ризику                            | Тип фейлу            | Вплив      | Пріоритет |
| ---- | --------------------------------------- | -------------------- | ---------- | --------- |
| R-01 | Retrieval miss                          | retriever / embedder | *high*     | *P2*      |
| R-02 | Fabrication                             | LLM                  | *critical* | *P1*      |
| R-03 | Context neglect / injection ignoring KB | LLM                  | critical   | P1        |
| R-04 | Entity confusion / conflicting facts    | chunking / context   | high       | P3        |
| R-05 | Cross-lingual EN↔UA consistency         | retriever/LLM        | medium     | P4        |
| R-06 | Prompt injection                        | retriever/LLM        | high       | P3        |




## 4. Підхід до тестування

- Типи перевірок: *функціональні / метричні / adversarial / регресійні*
- **Техніки дизайну тестів** (як добирав кейси): *еквівалентні класи / межові значення / негативні / adversarial / комбінаторика*
- Метрики та **чому саме вони** (прив'яжи кожну до ризику):  
//TODO: *...*
- Інструменти: *pytest / DeepEval / Ragas /LangChain / LangSmith*



## 4а. Traceability (ризик → кейси → результат)

Ключова QA-дисципліна: кожен ризик має бути простежуваним до тестів і до вердикту.


| risk_id | Кейси (id з датасету)                  | Метрика/перевірка     | Статус (pass/fail) | Дефект (ID зі звіту) |
| ------- | -------------------------------------- | --------------------- | ------------------ | -------------------- |
| R-01    | Q1, Q3, Q6–Q10, Q13–Q15, Q19, Q20, Q25 | *<метрика/перевірка>* | *pass/fail*        | *D-0X або —*         |
| R-02    | Q4, Q26–Q30, Q35                       |                       |                    |                      |
| R-03    | Q32, Q33                               |                       |                    |                      |
| R-04    | Q21–Q24                                |                       |                    |                      |
| R-05    | Q2, Q5, Q11, Q12, Q16–Q18              |                       |                    |                      |
| R-06    | Q31, Q34                               |                       |                    |                      |




## 5. Обробка недетермінізму

- Як фіксуємо відтворюваність: temperature не застосовна / greedy (do_sample=False).
- Скільки прогонів на кейс і який поріг pass-rate: // TODO: *напр. 5 прогонів, поріг 0.8*



## 6. Критерії проходження/непроходження та Definition of Done

- Кейс вважається пройденим, якщо: *...*  
*//TODO: fulfill ...*
- Поріг для метрик:  
**Precision@K** ≥ **0.8**  
**Recall@K** ≥ **0.8**  
**MRR** ≥ **0.8**  
**NDGS** ≥ 0.8  
**Context/generation metrics:**  
FAITHFULNESS  ≥  0.8    
ANSWER_RELEVANCE ≥ 0.8  
CONTEXT_CORRECTNESS ≥ 0.8  
CONTEXT_PRECISIO ≥ 0.8

     ANSWER_CORRECTNESS ≥ 0.8

- **Entry criteria** : *SUT доступний, датасет ≥30, ризики визначені, SLA for metrics are set, tools for testing is selected*
- **Exit criteria / DoD** : *run_eval зелений офлайн; усі P1-ризики покриті кейсами; дефекти задокументовані з severity й root cause; звіт і traceability заповнені*



## 7. Дані

- Джерело eval-датасету: *data/eval_dataset.json*
- Розподіл за категоріями (happy / edge / negative / adversarial):  
//TODO: fill with correct quantity  
*-* happy : test cases   
- edge : test cases  
- negative : test cases  
- adversarial: test cases



## 8. Ризики самого процесу тестування й обмеження

//TODO: 

- situation when doc gets re-chunked after an update, but ground dataset doesn't . It can provoke silent failures and can be quite expensive
- *напр. суддя-LLM може бути упередженим;* 
- *обмежений розмір вибірки*

