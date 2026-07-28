# CI/CD for Track B RAG

## 1. Тригери

CI/CD-конвеєр для цього SUT запускається на:

- `pull_request` (не-draft) — **pre-merge gate**, блокує merge
- `push` у `main` після merge — **post-merge gate**, перевіряє `main`, merge уже відбувся
- nightly-розкладом (`schedule`)
- on-demand перед релізом (manual `workflow_dispatch` / release tag)

Найчастіше для цього RAG-SUT змінюються код адаптера, eval-тести, golden-set і збережені генерації. Саме тому будь-яка зміна в `src/system_under_test.py`, `tests/test_eval.py`, `tests/test_redteam.py` або `data/eval_dataset.jsonl` повинна запускати перевірки, навіть якщо база знань чи модель формально не змінювались.

Окремо зміни в `outputs/generations.json` теж є тригером, бо в цьому репозиторії саме цей артефакт є входом для офлайн-оцінювання. Якщо змінюється лише документація, наприклад `reports/results.md`, повний CI не потрібен.

## 2. Яруси (Stages)

### PR stage

На кожен PR запускаються лише швидкі й детерміновані перевірки:

- `tests/test_functional.py`
- retrieval-частину з `tests/test_eval.py` для test cases з severity = critical
- `tests/test_redteam.py::test_no_secret_leak`

**Що:** functional + retrieval на committed `generations.json` + `test_no_secret_leak`.

**Час:** ~2–5 хв.  
**Блокує:** так.
**Бюджет:** ≤ $1 (у цьому проєкті фактично $0 — офлайн по JSON, без API).

Командa:

- python -m pytest -m "smoke and severity_critical" -v



### Post-merge gate (push to `main`)

Запускається **після** merge у `main` (`on: push`). Merge вже відбувся; цей job не може його заблокувати — лише підтверджує якість на `main` і сигналізує про регресію (failed check, email/issue).

**Що:** повний офлайн-eval на всіх 37 кейсах із `outputs/generations.json` (functional + retrieval-метрики); red-team safety (`test_no_secret_leak`); retrieval-звіт у `reports/`.

**Час:** ~10–20 хв (без GPU-judge).  
**Блокує merge:** ні (merge уже завершено) → **hard fail на `main` + alert** (GitHub check, email за наявності SMTP).

Команди (як у `.github/workflows/ai-pr-gate.yml`, job `merge-to-main`):

- `python -m pytest tests/test_functional.py -v`
- `python -m pytest tests/test_eval.py -k "retrieval and not ragas" -v`
- `python src/report_retrieval.py --k 3`
- `python -m pytest tests/test_redteam.py::test_no_secret_leak -v`

Еквівалент однією командою локально: `python -m pytest -m smoke -v`.


### Nightly — не блокує merge, лише alert

**Що:**
- functional: all functional tests
- eval : all eval tests with LLM as a judge
- red-team: all red team tests
- сегментна регресія: окремо EN vs UA, happy vs negative/adversarial, risk_id R-01…R-06:
Example for lang="en":
* python -m pytest -m lang_en -v
- повний adversarial / `test_safe_refusal` + ASR
- Ragas LLM-as-judge (`test_ragas_qwen3_judge`), якщо є GPU-runner

**Час:** ~30–90 хв (залежить від Qwen3-8B).
**Блокує:** ні → **alert** (GitHub notification / issue).
**Бюджет:** ≤ $10–50 за ніч; для 37×5 метрик ≈ 185 judge-evaluations; у локальному варіанті — переважно GPU-час, не token bill.


### Release — блокує деплой

**Що:**

1. `TRACK=B python src/generate.py --n-runs 5` (свіжі генерації)
2. повний eval + Ragas + red-team
3. human review дефектів / release notes
  post-deploy monitoring

**Час:** ~1–3 год (генерація + judge + review).
**Блокує:** так.
**Бюджет:** ≤ $100 (лекція); у локальному варіанті — переважно GPU-час, не token bill.

---

## 3. Гейти (Quality Gates)

Quality gate — **договір команди про мінімально прийнятну якість**.
Пороги калібруються на historical data проєкту: зафіксовані `≥ 0.8` у `test_strategy.md` i `≥ 0.7 для tests/test_eval.py`, спостережені Ragas-means нижчі (`faithfulness ≈ 0.50`, `answer_correctness ≈ 0.35`), ASR інколи > `0.10`. Тому merge-gates жорсткі лише там, де перевірки швидкі й стабільні.s

### Merge blockers (лише PR gate)

Pre-merge перевірки на `pull_request`. Post-merge job на `push` до `main` використовує ті самі пороги, але **не блокує merge** — див. §2 Post-merge gate.


| Gate              | Умова                                                       | Джерело                                                              |
| ----------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| Data / schema     | `test_functional` зелений; датасет ≥ 30                     | `tests/test_functional.py`                                           |
| Retrieval suite   | Hit pass-rate, MRR, Recall@K, NDCG@K, Precision@K ≥ **0.8** | `test_strategy.md`                                                   |
| Safety (critical) | `test_no_secret_leak` — **0** critical leak failures        | `tests/test_redteam.py` + лекція («0 critical adversarial failures») |

### Warnings на PR → blockers на Release / Nightly alert


| Метрика                            | PR   | Release blocker                                                                                         | Примітка            |
| ---------------------------------- | ---- | ------------------------------------------------------------------------------------------------------- | ------------------- |
| faithfulness                       | warn | ≥ **0.70** (калібровано вниз від 0.8 через historical ~0.50; ціль підняти до 0.85 як у прикладі лекції) | Ragas + Qwen3-8B    |
| answer_relevancy                   | warn | ≥ 0.70                                                                                                  |                     |
| answer_correctness                 | warn | ≥ 0.70                                                                                                  | historical ~0.35    |
| context_precision / context_recall | warn | ≥ 0.70                                                                                                  |                     |
| `test_safe_refusal` / ASR          | warn | ASR ≤ **0.10** (`SLA_MAX_ASR`)                                                                          | R-02                |
| Latency P95                        | —    | **P95 < 2000 ms**                                                                                       | міряти після деплою |
| Cost / query                       | —    | **< $0.05** (якщо з’явиться платний API)                                                                | лекція              |


Відомі дефекти (напр. conflicting Free storage d2 vs d1, brand hallucination) документуються в `reports/results.md` і, за потреби, `@pytest.mark.xfail` — вони **видимі**, але не маскуються без ID дефекту.

---

## 4. Статистична перевірка

AI-тести можуть бути **probabilistic**. У цьому SUT генерація greedy (`do_sample=False`), тож retrieval і regex-oracles майже детерміновані: падіння = реальна регресія, не шум.

Для відділення регресії від шуму:

1. **Повторні прогони:** `--n-runs 5` перед релізом; pass-rate по кейсу ≥ **0.8**.
2. **Мінімальний розмір ефекту:** Δ mean-метрики < **0.03** між сусідніми прогонами → не rollback / не hard-fail без review (узгоджено з ідеєю «F1 ≥ baseline − 0.02» з лекції).
3. **Bootstrap** на nightly для `faithfulness` і `answer_correctness` по 37 кейсах: якщо 95% CI різниці повністю < 0 **і** |Δ| ≥ 0.03 → реальна деградація → alert.
4. **Сегменти:** окремо дивитись EN/UA і risk_id, щоб не сховати регресію в середньому по всьому сету.

---

## 5. Вартість і час виконання

Бюджети, адаптовані під проєкт:


| Етап                    | Час (цей проєкт) | Бюджет лекції | Як контролювати                                                  |
| ----------------------- | ---------------- | ------------- | ---------------------------------------------------------------- |
| PR (pre-merge)          | 2–5 хв           | ≤ $1          | офлайн JSON, smoke + severity_critical, path filters             |
| Post-merge (push main)  | 10–20 хв         | —             | повний offline suite на `main`, без Ragas; alert, не блок merge  |
| Nightly                 | 30–90 хв         | ≤ $10–50      | judge лише тут; кеш моделей HF; без зайвого regenerate           |
| Release                 | 1–3 год          | ≤ $100        | один повний regenerate + human review + post-deploy monitoring   |


Оптимізації:

- path filters (не ганяти gate на чисті docs)
- committed `outputs/generations.json` як кеш генерації
- Ragas / `--n-runs 5` не на кожен PR
- кеш ідентичних промптів/ембедингів між запусками
- subset тест кейсiв на PR замість повних 37×judge

Якби nightly judge був платним API: ~185 evaluations → орієнтовно **до кількох доларів за ніч**, у межах nightly-бюджету лекції.

---

## 6. Робота із секретами

- Зберігання: **GitHub Actions Secrets** (CI) і локальний `.env`.
- Доступ: власник репозиторію / maintainers pipeline.
- Заборона: ключі в `outputs/`, ноутбуках, логах (`echo` секретів), артефактах workflow.
- Контроль для цього проєкту: `.gitignore` → `python check_submission.py` → `test_no_secret_leak`.
- Зараз SUT локальний (без ключів); policy вже готова до опційного `OPENAI_API_KEY` / хмарного judge.

---

## 7. Сигналізація

Результати PR-пайплайна мають бути видимі прямо в GitHub Checks, щоб автор PR одразу бачив чи пройшов gate. Nightly і pre-release я б дублювала ще й у повідомлення на email для власника репозиторію та для людини, яка зробила змiни, бо саме вони приймають release decision.

Після кожного запуску мають формуватись артефакти:

- `outputs/generations.json` для відтворюваності
- `outputs/rag_evaluation_results.json`
- `outputs/ragas_metrics.log`
- job summary (pass-rate, retrieval means, ASR, latency - для release job)

Для production observability додатково логувати: `request_id`, `input`, `output`, `model_version`, `latency`, `token_count`, `temperature`/`top_p` , `retrieved_chunks` **/** `sources`, `user_feedback`. Трейсинг ланцюга retrieve→generate — LangSmith.

---

## 8. Після деплою

### Guardrail-метрики (2–3+)

1. **Safe-refusal rate** на synthetic production-check запитах ≥ **0.90** (еквівалент ASR ≤ 0.10).
2. **Share of answers with non-empty** `sources` ≥ **0.95** (контракт grounding).
3. **Unsupported / unfaithful rate** (відповідь суперечить `retrieved_chunks`) ≤ **0.10**.
4. **latency P95 < 2000 ms**, error rate.

Між релізами: сегменти EN/UA, топ risk_id (R-02 fabrication, R-01 retrieval miss), cost per query, user_feedback.

### Rollback без canary (конкретна політика)

Я свідомо **не використовую canary deployment** для цього проєкту. Натомість після повного деплою запускається короткий набір synthetic production-check запитів і вмикається посилений моніторинг першої години: error rate, latency P95, safe-refusal, unsupported rate, feedback.

Перед повним увімкненням нової версії я покладаюсь на:

1. повний pre-release eval
2. human review release-кандидата
3. за можливості shadow evaluation на staging або на копії production-трафіку без відповіді користувачу

**Автоматичний rollback**, якщо в **двох послідовних 15-хвилинних вікнах** після деплою:

- safe-refusal rate < **0.90**, або
- unsupported-answer rate > **0.10**, або
- P95 latency > **2000 ms** протягом обох вікон **і** error rate зріс > **2×** baseline

У такому разі система повертається на попередній стабільний набір артефактів: код застосунку, модель/промпт і snapshot knowledge-base / retrieval-індексу.

---
## Компроміси

### 1. Немає повного eval (37 + Ragas) на кожен PR

**Чому:** лекція радить на PR лише smoke 10–20 кейсів за 2–5 хв і ≤ $1; повний judge на Qwen3-8B ламає цей SLO.
**Ціна:** semantic регресія може пройти в main і виявитись лише на nightly/release.
**Ризик:** пізніше виявлення fabrication / faithfulness drop.

### 2. LLM-as-a-Judge не є merge blocker на PR

**Чому:** probabilistic, довгий, залежить від GPU; на PR блокуємо детерміновані retrieval + safety leak.
**Ціна:** слабший semantic контроль на швидкому шляху.
**Ризик:** «зелений PR» при падінні faithfulness (як у historical ~0.50).

### 3. Я свідомо відмовляюсь від canary deployment

**Чому:** у цьому проєкті немає окремої інфраструктури для поступового rollout по відсотках трафіку, а сам SUT оцінюється переважно через офлайн-eval і synthetic production-check.

**Що замість:** повний pre-release eval, короткий post-deploy synthetic check, чітке rollback-правило і, за можливості, shadow evaluation.

**Ціна:** розрив між staging-quality і live behavior.

**Ризик:** частина проблем буде виявлена вже після повного ввімкнення нової версії. Мітигація — жорсткіший pre-release gate і автоматичний rollback за конкретними production-метриками.

---
## Підсумок

Найслабший етап CI/CD flow — це **nightly/release рівнi (Ragas + Qwen3-8B)** іs post-deploy спостережуваність. Він дає найцінніший сигнал по faithfulness/correctness, але найдорожчий за часом і найменш придатний для PR-feedback, а без canary я ще сильніше покладаюся на якістsь pre-release eval і швидкий rollback. У проєкті retrieval і deterministic safety добре лягають на AI CI/CD, а judge і production-моніторинг відстають.

---
## AI-usage disclosure

AI допоміг мені з механікою оформлення документа, структуруванням CI/CD-опису під уже наявні артефакти проєкту та ескізом workflow-файлу.