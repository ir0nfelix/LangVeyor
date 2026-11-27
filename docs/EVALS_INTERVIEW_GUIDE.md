# Инженерный гайд по Evals мультиагентных систем (на примере LangVeyor MES RCA)

> Этот документ написан как практическая шпаргалка уровня **Tech Lead / AI Architect** для системного проектирования оценки (Evaluations) LLM-агентов и уверенного прохождения технических собеседований. Все примеры, формулы и метрики привязаны к реальному коду этого репозитория.

---

## 1. Зачем нужны Evals, если уже есть Unit- и Integration-тесты

Классическое заблуждение: *«Мы покрыли код `pytest` — зачем ещё какие-то evals?»*

1. **Детерминированная логика vs стохастическая система:**
   - Юнит-тесты (`tests/test_alg001.py`, `tests/test_db_genealogy.py`) проверяют бинарный инвариант: функция `check_stop_crane` либо вернула `DEFECT_RATE_EXCEEDED`, либо упала.
   - Мультиагентный граф с живой LLM не даёт бинарной гарантии: модель может верно вызвать `get_item_genealogy`, но **забыть** вызвать `find_affected_products_by_lot`, придумать цифру ёмкости «по памяти» (галлюцинация) или уйти в DuckDuckGo при наличии явного брака в БД.
2. **Тихая деградация (Silent Regression):**
   - Вы подкрутили 2 строчки системного промпта в `investigator_node` (`src/agent/graph.py`), чтобы улучшить объяснение по моторам `RS550-24V` — и незаметно сломали вызов `get_instrument_channel_anomalies` по стенду `STAND-03`.
   - Провайдер OpenRouter обновил веса модели или квантование, из-за чего `structured_output` начал чаще ошибаться в `affected_cars`.
   - Без автоматических Evals на Golden Dataset (`evals/golden_cases.json`) команда узнаёт об этом только от возмущённых пользователей в продакшене.

---

## 2. Пирамида Evals, реализованная в этом проекте

В `evals/metrics.py` и `evals/run_evals.py` построена трёхуровневая пирамида оценки (от самых дешёвых и быстрых проверок к семантическим и бизнес-метрикам):

```text
                / \
               /   \      Уровень 3: Бизнес и Операции
              / ROI \     (Token Cost, P95 Latency, Auditor Recovery Rate)
             /-------\
            /         \   Уровень 2: LLM-as-a-Judge (опционально в --mode live)
           /  Semantic \  (Оценка полноты и логики вердикта моделью-судьёй)
          /-------------\
         /               \ Уровень 1: Детерминированные Code-Based Evals
        / Code Assertions \ (Tool Trajectory, Recall F1, Web FPR, Groundedness)
       /___________________\
```

### Уровень 1: Детерминированные Code-Based Evals (быстрые, 0 токенов, запускаются в CI)
Реализованы в `evals/metrics.py`:
1. **Tool Trajectory Score (`compute_tool_trajectory_score`):**
   - Проверяет, вызвал ли `investigator_node` все обязательные SQL-тулы для данной ветки BPMN (`expected_required_tools`), и штрафует за лишние зацикленные вызовы (`max_allowed_calls = 8`).
   - *Почему это важно:* если агент угадал ответ без чтения БД, это чистая галлюцинация; если крутился 15 шагов — сожжены деньги и время.
2. **Recall Completeness (`compute_recall_prf1`):**
   - Для кейса `GC-01-LOT-DEFECT-RECALL` (`CAR-2026-0412` из партии `LOT-BAT-2026-014`) вычисляет **Precision / Recall / F1** предсказанного списка `final_verdict.affected_cars` против эталона в `evals/seed_manifest.json` (ровно 12 машинок).
   - *Критичность для бизнеса:* пропуск хотя бы 1 машинки (Recall < 1.0) в реальном парке аттракционов — это риск возгорания или поломки на трассе; лишняя машинка (Precision < 1.0) — лишние затраты на выезд механика.
3. **Action Match Score (`compute_action_match_score`):**
   - Проверяет **Precision / Recall / F1** предсказанных `proposed_actions[*].action_type` относительно `expected_action_types` из `evals/golden_cases.json` (`CREATE_SUPPORT_TICKET`, `CREATE_LOT_RECALL`, `CREATE_CALIBRATION_HOLD`, `ESCALATE_REPEAT_TICKET`, пустой список для кейса уточнения). Это не даёт агенту «угадать диагноз в тексте», но забыть заблокировать стенд или отозвать партию.
4. **Web-Routing Precision & FPR (`web_routing_fpr`, `web_routing_recall`):**
   - **Web-Routing FPR (False Positive Rate):** доля внутренних аномалий БД (`GC-01..04`, `GC-06`, `GC-08`), где граф **ошибочно** пошёл в интернет (`web_researcher_node`). Цель: **0%**.
   - **Web-Routing Recall:** доля «чистых» кейсов (`CAR-2026-0777` и др.), где `auditor_node` корректно вернул `NEED_WEB_RESEARCH`. Цель: **100%**.
5. **Numeric Faithfulness Guardrail (`compute_numeric_groundedness`):**
   - Это детерминированный **L1-guardrail числовой достоверности (numeric faithfulness)**: он извлекает числовые метрики из `final_verdict.confirmed_db_facts` (предварительно очищая UUID, ISO-таймстампы и серийные коды) и проверяет, что **каждое число** присутствует в сырых JSON-ответах тулов `evidence_log` (включая `min_value`/`max_value`/`nominal_value` допусков). При `require_facts=True` пустой `confirmed_db_facts` на внутренних аномалиях получает `0.0`.
   - *Важное ограничение:* эта метрика контролирует **только числа в `confirmed_db_facts`** (не текстовые формулировки причин и не `external_hypotheses`) и не заменяет полноценную NLI / semantic faithfulness проверку.

### Уровень 2: LLM-as-a-Judge (`_run_llm_judge` в `evals/run_evals.py --mode live --judge`)
Когда структура и цифры уже проверены Уровнем 1, LLM-судья оценивает семантическое качество диагноза (`0.0..1.0`) по структурированному эталону (`expected_root_cause_type`, `expected_entities`, `expected_action_types`, `description`):
- **Борьба с Position & Verbosity Bias:** в промпте судьи явно запрещено поощрять «воду» и длинный текст; оценивается строго фактическое совпадение первопричины, сущностей и действий. При парном сравнении версий промпта (A/B) кандидатов нужно менять местами (position swapping).
- **Полная кросс-семейственная изоляция (`AUDITOR` != `INVESTIGATOR` != `JUDGE`):** у нас `INVESTIGATOR` — это `deepseek/deepseek-v3.2`, `AUDITOR` — `openai/gpt-4.1-mini`, а для итоговой оценки отчётов в Evals выделена отдельная независимая роль **`OPENROUTER_MODEL_JUDGE`** (например, `anthropic/claude-haiku-4.5`). Использование отдельной модели-судьи (а не `auditor_model`) устраняет *self-preference bias* ещё и на этапе оценки итогового вердикта, гарантируя независимый учёт токенов, задержки и стоимости под ролью `JUDGE`.

### Уровень 3: Бизнес- и операционные метрики
- **Latency & Token Cost по ролям (`role_breakdown`):** через `include_raw=True` в `invoke_structured_with_usage` (`src/agent/llm.py`) мы фиксируем реальные `input_tokens`, `output_tokens`, задержку и USD-стоимость отдельно для `FAST`, `INVESTIGATOR`, `AUDITOR` и `JUDGE`. Это позволяет доказать экономику роутинга: дешёвая `FAST`-модель берёт на себя шумный веб-подграф и первичный intake, а дорогая reasoning-модель тратится только на расследование.
- **Auditor Catch & Recovery Rate + `auditor_unit_check`:** измеряет реальный ROI паттерна Reflection (`auditor_node`). В `--mode mock` на `GC-01` (`simulate_incomplete_draft_catch=True`) мы намеренно моделируем перехват неполного черновика. В `--mode live` сильная LLM обычно вызывает `find_affected_products_by_lot` с первого раза (поэтому живой кейс не штрафуется за отсутствие ревизии), а работоспособность Ревизора независимо проверяется детерминированным **`auditor_unit_check`** в конце прогона evals: в `auditor_model` передаётся заведомо неполный черновик без `affected_cars` при 20% браке партии и проверяется вердикт `REJECTED`.

### Недетерминизм живых LLM (`pass@k` vs `pass^k`) и Online Production Tracing
При оценке стохастической системы в `--mode live` разового прогона недостаточно — на собеседовании важно разделять две метрики повторяемости:
- **`pass@k`** — вероятность того, что **хотя бы одна** из `k` попыток завершится успехом (оптимистичный потолок возможностей модели/графа, полезный для R&D).
- **`pass^k` (`pass-hat-k`)** — вероятность того, что **все `k` независимых прогонов подряд** пройдут Golden Dataset без единого сбоя. Для индустриальной MES/RCA-системы критичен именно высокий **`pass^k`** (консистентность от смены к смене при `temperature=0` и ретраях `function_calling`).
- **Online Production Tracing (Langfuse / OpenTelemetry):** помимо офлайн-прогонов `evals/run_evals.py`, в продакшене каждый вызов графа (трассы узлов, латентность SQL-тулов, fallback-повторы `invoke_structured_with_usage`, `parsing_error`, стоимость по ролям и решения HITL `approved=True/False`) отправляется асинхронно в **Langfuse** или **OpenTelemetry**-коллектор. Это связывает оффлайн Golden Dataset с реальным продакшен-дрейфом без задержки основного ответа оператору.

---

## 3. Как проектировать Golden Dataset через «посаженные аномалии»

Ждать 6 месяцев ручной разметки реальных инцидентов в продакшене — антипаттерн. В `src/db/seed.py` применена техника **Synthetic Ground Truth (посаженные аномалии)**:
1. Детерминированный генератор (`random.Random(20260101)` + `uuid5`) за 5 секунд засеивает 12 месяцев реалистичного шума (~451 машинка, ~18 партий, ~4800+ операций).
2. Внутри массива данных детерминированно заложены 5 эталонных первопричин разных классов (`LOT_DEFECT`, `INSTRUMENT_CHANNEL_DRIFT`, `OPERATOR_QUALITY_DROP`, `REPEAT_FAILURE`, `DB_CLEAN_EXTERNAL_SUSPECTED`) + граничные кейсы (нет серийника, несуществующий серийник, пограничный шум).
3. Сидер автоматически экспортирует истину (`evals/seed_manifest.json`), от которой питаются и тесты `pytest`, и `evals/run_evals.py`. При перегенерации данных ground truth никогда не рассинхронизируется!

---

## 4. Почему `Web-Routing FPR` критичен для продакшена

Представьте: у машинки `CAR-2026-0412` в БД на заводе зафиксирован 20% брак партии АКБ от поставщика, но агент из-за плохого промпта решил поискать в интернете и заявил механику: *«Возможно, это мороз на улице»*.
- **Последствие 1 (Упущенный отзыв):** 11 остальных машинок из битой партии остаются в парке, пока не выйдут из строя.
- **Последствие 2 (Галлюцинаторный шум):** веб-сниппеты загрязняют контекст, смешиваются с фактами и подрывают доверие инженеров к системе.
- **Как мы это победили архитектурно:**
  1. Условный переход в `web_researcher_node` принадлежит **только** `auditor_node` (`NEED_WEB_RESEARCH`), а не `investigator_node`.
  2. `web_researcher_node` вынесен в изолированный **Subgraph** (Context Hygiene): сырой веб-мусор из DuckDuckGo не попадает в главный `messages`.
  3. В `AgentState` и `FinalVerdict` поля `confirmed_db_facts` и `external_hypotheses` разделены жёсткой схемой Pydantic.

---

## 5. Как тестировать `interrupt()` и чекпоинты LangGraph

1. **Никогда не мокай checkpointer на `MemorySaver` в финальных тестах, если в проде Postgres:** в `tests/test_graph_e2e.py` мы поднимаем реальный `AsyncPostgresSaver` поверх `psycopg_pool.AsyncConnectionPool(autocommit=True)` и делаем `await checkpointer.setup()`.
2. **Проверяй две фазы `interrupt()`:**
   - **Фаза ДО `resume`:** граф доходит до `action_node`, останавливается с `snapshot.tasks[...].interrupts`, а в PostgreSQL `support_tickets` и `recall_notices` содержат **0 новых строк** (гарантия отсутствия побочных эффектов до решения человека!).
   - **Фаза ПОСЛЕ `Command(resume=...)`:** проверяем оба исхода: при `approved=True` в одной ACID-транзакции создаются записи в БД; при `approved=False` узел корректно завершается с 0 записей в БД.
3. **Идемпотентность `action_node`:** при `Command(resume=...)` LangGraph перезапускает весь `action_node` с первой строчки до `interrupt()`. Любая запись в БД до вызова `interrupt()` продублировалась бы или упала бы!

---

## 6. Offline CI Evals vs Online Production Evals

| Аспект | Offline Evals (`evals/run_evals.py`) | Online Evals (Production) |
| :--- | :--- | :--- |
| **Когда запускается** | На каждом PR (`--mode mock` в CI за 5 сек) и перед релизом промптов/моделей (`--mode live`) | Непрерывно асинхронно на реальном трафике (shadow / sampling 5–10% тикетов) |
| **Данные** | `evals/golden_cases.json` + детерминированный сидер | Реальные обращения из парков аттракционов |
| **Ground Truth** | Заведомо известен (`seed_manifest.json`) | Неизвестен сразу; поступает с задержкой (Implicit Feedback: принял ли оператор HITL-черновик без правок, закрыт ли тикет без повторного возврата) |
| **Ключевые сигналы** | Regression guard: Pass Rate, Recall F1 = 1.0, Web FPR = 0% | Дрейф данных (новые модели АКБ/моторов), процент отклонений HITL (`approved=False`), доля `ESCALATED` и повторных тикетов |

---

## 7. Типовые каверзные вопросы на собеседовании и сильные ответы

### Вопрос 1: *«Почему вы запретили агенту Text-to-SQL и написали 7 жёстких SQL-тулов? Разве Text-to-SQL не гибче?»*
**Сильный ответ:**
На реальном производстве Text-to-SQL даёт три фатальных риска: (1) безопасность и случайные тяжёлые `JOIN` без индексов на таблице замеров; (2) неверная семантика бизнес-метрик — LLM регулярно путает `route_sheets.status` с `hold_reason` или забывает `WITH RECURSIVE` ограничение глубины и `link_type` (`SPLIT` vs `ASSEMBLED_INTO`); (3) **невозможность гарантировать повторяемость в evals**. Типизированные тулы в `src/agent/tools.py` (`query_item_genealogy`, `query_affected_products_by_lot`, `query_instrument_channel_anomalies`) инкапсулируют проверенные `WITH RECURSIVE` CTE и оконные агрегации ±14 дней, возвращая детерминированный JSON для `evidence_log`.

### Вопрос 2: *«Зачем нужен отдельный `auditor_node` на другой модели? Разве нельзя просто сделать длинный промпт у `investigator_node`?»*
**Сильный ответ:**
Один вызов с длинным промптом страдает от эффекта «замыливания внимания» (Lost-in-the-Middle) и самоподтверждения: модель, которая сгенерировала неполный черновик (например, забыла вызвать `find_affected_products_by_lot` при 20% браке партии), склонна оправдывать свой же пропуск. Паттерн Reflection с отдельным `auditor_node` на **другой семье моделей** (`openai/gpt-4.1-mini` против `deepseek/deepseek-v3.2`) работает как независимый код-ревью: Ревизор не видит внутренних рассуждений Следователя, а сверяет только `InvestigationDraft` против сырого `evidence_log` и ветки `BPMN`. На кейсе `GC-01` наш `Auditor Catch & Recovery Rate` доказывает, что Ревизор стабильно ловит пропущенный Recall и доводит F1 до `1.0`. Предохранитель `max_revisions = 2` гарантирует отсутствие бесконечного цикла и предсказуемый бюджет токенов.

### Вопрос 3: *«Как вы защитили главный контекст агента от переполнения сырым текстом из интернета и документов?»*
**Сильный ответ:**
Двумя механизмами. Во-первых, **Context Hygiene Subgraph** в `src/agent/web_research.py`: `web_researcher_node` — это вложенный `StateGraph` со своим `WebResearchState`. Он формулирует 2–3 запроса, выгружает сырые сниппеты DuckDuckGo, сжимает их внутри себя моделью `FAST` и возвращает в родительский `AgentState` только компактный список из 2–3 `ExternalHypothesis`. Во-вторых, для процессов мы **не режем BPMN XML на чанки в Qdrant** (`src/rag/bpmn_parser.py` парсит XML в `networkx.DiGraph` и вырезает только ветку конкретного симптома), а в Qdrant + BM25 RRF индексируются исключительно инженерные SOP-регламенты.

### Вопрос 4: *«Как вы гарантируете, что агент не перепутает предположения из интернета с фактами из базы данных?»*
**Сильный ответ:**
Строгой структурной изоляцией на уровне схемы состояния (`AgentState` в `src/agent/state.py`), маршрутизации и evals:
1. В `FinalVerdict` поля `confirmed_db_facts` и `external_hypotheses` разделены физически, а в `verdict_node` стоит кодовый инвариант: `verdict.external_hypotheses = list(ext_hyps)`.
2. `web_researcher_node` недостижим напрямую из `investigator_node` — туда пускает только `auditor_node` со статусом `NEED_WEB_RESEARCH` после проверки, что все обязательные SQL-тулы отработали и БД чиста.
3. В `evals/metrics.py` метрика `compute_numeric_groundedness` проверяет каждое число в `confirmed_db_facts` по `evidence_log`, а `Web-Routing FPR` штрафует за любой поход в интернет при наличии аномалии в БД.

### Вопрос 5: *«Какие подводные камни есть у LangGraph `interrupt()` в продакшене и как вы их избежали?»*
**Сильный ответ:**
В LangGraph при возобновлении через `Command(resume=...)` узел `action_node` **перезапускается с самого начала функции**. Поэтому:
1. Нельзя оборачивать `interrupt()` в `try/except` — иначе перехватится внутреннее исключение `GraphInterrupt` и пауза сломается.
2. Нельзя делать никаких записей в БД (`INSERT`/`UPDATE`) до `interrupt()` — при возобновлении запись выполнится дважды.
3. Аргумент `interrupt(payload)` должен быть чисто JSON-сериализуемым для `AsyncPostgresSaver`.
В `src/agent/graph.py` весь блок `async with session_factory() as session: async with session.begin(): ...` находится строго **после** `interrupt(hitl_payload)`, что подтверждено интеграционным тестом `test_graph_e2e_reflection_recall_and_hitl_postgres_commit` (проверка 0 строк в `support_tickets`/`recall_notices` в момент паузы и ACID-запись только после `Command(resume={"approved": True})`).

### Вопрос 6: *«Зачем в гибридном RAG использовать одновременно `cointegrated/LaBSE-en-ru` (Qdrant) и `BM25` с `RRF`?»*
**Сильный ответ:**
В промышленных регламентах одновременно встречаются семантические описания на русском языке (*«батарея быстро разряжается на морозе»*) и точные буквенно-цифровые коды компонентов, стендов и гардов (`JK-BMS-6S`, `RS550-24V`, `STAND-03`, `G-BAT-CAP`, `R_ac`). Dense-эмбеддинги (`LaBSE-en-ru`) отлично ловят русско-английскую семантику, но размывают точные артикулы вроде `STAND-03` vs `STAND-01`. Лексический `BM25Okapi` с нашей двуязычной токенизацией `tokenize_bilingual` (`src/rag/hybrid_retriever.py`) точно находит артикулы и числовые допуски, а **Reciprocal Rank Fusion (RRF)** стабильно объединяет ранги без необходимости калибровать несопоставимые шкалы косинусного сходства и BM25-скоров.

### Вопрос 7: *«Как детерминированно тестировать LLM-пайплайн в CI без сжигания API-квоты и флакающих тестов?»*
**Сильный ответ:**
Мы разделяем **проверку оркестрации** и **проверку промптов/весов модели**. В `src/agent/llm.py` и `src/agent/mock_models.py` реализован `ScriptedChatModel(BaseChatModel)`, поддерживающий реальные `bind_tools` и `with_structured_output(..., include_raw=True)`. В `--mode mock` и `pytest` запускается **полный реальный граф LangGraph** со всеми настоящими SQL-тулами против PostgreSQL 17, настоящим Qdrant Hybrid RAG, настоящим `AsyncPostgresSaver` и настоящим `interrupt()`. Меняется только слой генерации ответов LLM. А `--mode live` прогоняется по расписанию или перед релизом промптов/сменой слагов моделей в `.env`.
