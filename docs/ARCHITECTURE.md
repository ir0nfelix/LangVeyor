# Архитектура LangVeyor

Проект построен из двух связанных слоёв:

1. **Ядро конвейерного учёта (`PostgreSQL 17` + `SQLAlchemy 2.1 Async`)** — конечный автомат техкарт с пооперационными допусками, метрология стендов, сквозная генеалогия партий и изделий, а также потоковый выборочный контроль с автоматическим стоп-краном (`ALG-001`).
2. **Чат-агент расследования инцидентов (`LangGraph` + `BPMN 2.0` + `Hybrid RAG`)** — оркестратор техподдержки, который вырезает нужную ветку регламента из BPMN-схемы, проверяет историю сборки через 7 типизированных инструментов, сверяет черновик диагноза через независимую модель-ревизора и фиксирует решения в БД только после подтверждения оператора (HITL).

---

## 1. База данных и конечный автомат конвейера (`PostgreSQL 17`)

### 1.1. Инженерные соглашения схемы
- **Перечисления без `CREATE TYPE`:** статусы и типы связей хранятся как `VARCHAR(...)` с `CHECK`-ограничениями в PostgreSQL и зеркалируются в Python через `StrEnum` — миграции Alembic накатываются без блокировок и пересборки enum-типов.
- **Время и ключи:** все временные метки хранятся в `TIMESTAMPTZ` (`clock_timestamp()`). Первичные ключи — `UUID` (`gen_random_uuid()`), за исключением высокочастотного журнала замеров `step_measurements`, где используется `BIGINT GENERATED ALWAYS AS IDENTITY`.
- **Защита истории и именование:** кадровые и производственные записи не удаляются каскадно (`ON DELETE RESTRICT` в журналах исполнения), а все ограничения в БД следуют единой конвенции `pk_`, `uq_`, `fk_`, `idx_`, `chk_`.

### 1.2. Конфигуратор маршрутов, гардов и оборудования
- **Техкарты как DAG (`tech_maps`, `tech_map_nodes`, `tech_map_transitions`):** маршрут описывается графом узлов и типизированных переходов (`NORMAL`, `REWORK`, `DEFECT`). При активации проверяется ацикличность основного потока (алгоритм Кана).
- **Целостность на уровне составных ключей:** внешние ключи `(tech_map_id, from_node_id)` и `(node_id, measurement_guard_id)` на уровне PostgreSQL исключают привязку перехода или допуска к узлу чужой техкарты.
- **Правило «двух ключей» и иммутабельность:** технолог утверждает техкарту (`is_approved`), начальник участка активирует её (`is_active`), причём БД проверяет `approved_by_id != activated_by_id`. Утверждённая карта не редактируется на месте — изменения вносятся через создание новой версии (Copy-on-Write).
- **Область выполнения (`execution_scope`):** каждый узел техкарты помечен как `'LOT'` (партионная операция) или `'ITEM'` (поштучная операция).
- **Гарды (`tech_map_node_guards`) и метрология (`instruments`):** на узлах задаются числовые допуски, булевы чеки и регулярные выражения для штрихкодов. Справочник `instruments` хранит стенды и инструмент (зарядные стенды №1–3 по 16 каналов, измерители $R_{\mathrm{ac}}$, динамометрические ключи) вместе с датой следующей поверки `calibration_due_at`.

### 1.3. Производственные маршруты полигона (детские электромашинки)

1. **`TM-BAT-6S` — Приёмка и зарядка АКБ 6S 35 А·ч (`LOT → SPLIT → ITEM`):**
   - Входной контроль партии (`LOT`, `OP-LOT-RECEIVE`).
   - Расщепление партии на серийные батареи (`SPLIT`, `OP-LOT-SPLIT` — порождает связи `route_sheet_links.link_type = 'SPLIT'`).
   - Визуальный осмотр корпуса и клемм (`ITEM`, `OP-BAT-VISUAL`).
   - Зарядка и балансировка на многоканальном стенде `STAND-01..03`, каналы `1..16` (`ITEM`, `OP-BAT-CHARGE`).
   - Выходные электрические испытания (`ITEM`: напряжение $21.6\text{--}25.2\,\text{В}$, ёмкость $33.0\text{--}37.5\,\text{А}\cdot\text{ч}$, внутреннее сопротивление $8.0\text{--}18.0\,\text{мОм}$) с петлёй `REWORK`, линейным стоп-краном и выборочным контролем `ALG-001`.
2. **`TM-BODY` — Сборка кузова (`ITEM`):**
   - Контроль пластика и покраски → установка рулевого узла и сиденья → проверка крепежа и жгутов (с петлёй `REWORK`).
3. **`TM-CAR-FINAL` — Финальная сборка и обкатка (`ITEM`):**
   - Монтаж редуктора и DC-мотора `RS550-24V` с контролем момента затяжки (`8.0–12.0 Н·м`) → узел комплектации (связывает `TM-BODY` + `TM-BAT-6S` + контроллер `JK-BMS-6S` через `ASSEMBLED_INTO`) → тестовый заезд → финальный ОТК.

### 1.4. Схема исполнения и сквозная генеалогия (`WITH RECURSIVE`)

```text
[tech_maps] 1───* [route_sheets] 1───? [lot_metadata]
                        │
        ┌───────────────┼───────────────────────────────┐
        │               │                               │
        ▼               ▼                               ▼
[route_sheet_links]  [step_executions] 1───* [step_measurements]
 (SPLIT /             │    │
  ASSEMBLED_INTO)     ▼    ▼
                [users]  [instruments]
```

- **`route_sheets`** — единый курсор состояния для партий (`LOT`) и штучных изделий (`ITEM`). Поле `hold_reason` (nullable) отделено от рабочего `status` (`IN_PROGRESS`, `SPLIT_OPEN`, `SPLIT_CLOSED`, `COMPLETED`, `DEFECT`, `QUARANTINE`), чтобы блокировка партии не затирала технологический этап. Денормализованная индексированная ссылка `root_lot_id` даёт $O(1)$ проверку блокировки корневой партии и быстрый доступ к счётчикам `defect_count` и `processed_count`.
- **`lot_metadata`** — `1:1` анкета корневой партии: поставщик (`supplier_name`), номер накладной (`invoice_number`), дата прихода (`arrival_date`) и `component_specs JSONB` с BOM-составом (модель мотора, BMS, химия ячеек).
- **`route_sheet_links`** — направленный граф материальной генеалогии `source_rs_id -> target_rs_id` (`SPLIT` для $1 \to N$ расщепления партии и `ASSEMBLED_INTO` для $N \to 1$ сборки готовой машинки). Составные индексы `(source_rs_id, link_type)` и `(target_rs_id, link_type)` позволяют одним рекурсивным SQL-запросом (`WITH RECURSIVE`) пройти граф в обе стороны:
  - *назад (Genealogy)* — от сломанной машинки к кузову, батарее, стенду зарядки, оператору и поставщику ячеек;
  - *вперёд (Recall Analysis)* — от бракованной партии ко всем остальным машинкам, ушедшим в парки под отзыв.
- **`step_executions` и `step_measurements`** — append-only журналы прохождения операций (исполнитель, прибор, канал стенда, номер групповой загрузки `group_batch_code`, исход, применённый переход) и конкретных замеров по каждому гарду.
- **`sampling_node_states`** — персистентное состояние выборки `ALG-001` по ключу `(lot_rs_id, node_id)`: `m_rem`, `n_rem`, `w_rem`, `taken_in_window`, `escalations_used`.
- **`support_tickets` и `recall_notices`** — журналы обращений и отзывов, пополняемые только после подтверждения оператора.

---

## 2. Потоковый выборочный контроль и стоп-кран (`ALG-001`, `../src/mes/sampling.py`)

На участке лабораторного контроля решение об отборе изделия на стенд принимается в потоке без заранее известного списка серийников, в $O(1)$ памяти и под транзакционной блокировкой `SELECT ... FOR UPDATE`:

1. **Математика выборки (Algorithm S Кнута / Фан–Мюллер–Резуха, 1962):**
   Для партии объёмом $N$ и квоты $m$ формируется стратифицированное окно размером $w_{\mathrm{rem}} = \max(1, \lfloor n_{\mathrm{rem}} / m_{\mathrm{rem}} \rfloor)$ при $n_{\mathrm{rem}} > m_{\mathrm{rem}}$. Внутри окна (пока `taken_in_window == 0`) очередное изделие отбирается с вероятностью $p = 1 / w_{\mathrm{rem}}$, что гарантирует ровно одну лабораторную проверку на каждую страту. После отбора `taken_in_window` инкрементируется, `m_rem` уменьшается на 1, а остаток окна пропускается.
2. **Прикладной источник случайности (`HMAC-SHA256`):**
   В классическом алгоритме Кнута используется абстрактный генератор равномерных случайных чисел, но хранить и блокировать стейт PRNG в PostgreSQL при параллельных запросах с рабочих мест непрактично. Поэтому равномерное число $u \in [0, 1)$ вычисляется как чистая функция от тройки `(lot_rs_id, node_id, item_tracking_code)` и секретного ключа:

   $$
   u = \frac{\mathrm{int}(\mathrm{HMAC\_SHA256}(\mathrm{key}, \mathrm{lot\_rs\_id}:\mathrm{node\_id}:\mathrm{item\_code})[:8])}{2^{64}}
   $$

   Такой подход избавляет БД от хранения состояния генератора, делает выборку 100% воспроизводимой при сетевых ретраях или повторном прогоне симуляции и не позволяет оператору на линии заранее угадать, какая батарея пойдёт на стенд.
3. **Эскалация при браке и триада состояний (`n_rem` vs `m_rem`):**
   При обнаружении дефекта в выборке (до `max_escalations` раз) квота `m_rem` увеличивается на `escalation_sample_size`:
   - `n_rem < m_rem` — дефицит деталей на остатке партии: включается 100% сплошной контроль ($p = 1.0$) с фиксацией нехватки `shortfall = m_rem - n_rem` (`HOLD_INSUFFICIENT_SAMPLE`);
   - `n_rem == m_rem` — все оставшиеся изделия отбираются детерминированно ($p = 1.0$);
   - `n_rem > m_rem` — окно $w_{\mathrm{rem}}$ пересчитывается под возросшую квоту и продолжается равномерный отбор.
4. **Корректировка при недопоставке (`delta`):**
   Если при расщеплении партии фактически пришло меньше деталей на `delta`, остаток `n_rem` уменьшается (`n_rem <- max(0, n_rem - delta)`), а окно сжимается так, чтобы сохранить инвариант $w_{\mathrm{rem}} \le n_{\mathrm{rem}} - m_{\mathrm{rem}}$ (или сразу переходит в 100% контроль при $n_{\mathrm{rem}} \le m_{\mathrm{rem}}$).
5. **Линейный стоп-кран партии:**
   Партия автоматически уходит в блокировку `hold_reason = 'DEFECT_RATE_EXCEEDED'` (без затирания рабочего `status`), как только выполняется условие:

   $$\mathrm{defect\_count} \ge d_{\min} + R_{\mathrm{slope}} \times \mathrm{processed\_count}$$

   Это отсекает ложные срабатывания на первых 1–2 единицах и останавливает партию при статистически устойчивом всплеске брака.

---

## 3. База знаний: BPMN-процесс vs Гибридный RAG (`../src/rag`)

### 3.1. Процессный граф (`../src/rag/bpmn_parser.py`)
- **Никакой нарезки XML в векторную БД:** схема `../knowledge_base/support_process.bpmn` не индексируется в Qdrant. `BpmnWorkflowParser` парсит XML через `lxml.etree` в направленный граф `networkx.DiGraph`, сохраняя дорожки (`lane`), типы узлов (`startEvent`, `endEvent`, `userTask`, `serviceTask`, `exclusiveGateway`), имена инструментов из `serviceTask` и формальные выражения переходов `<bpmn:conditionExpression>`.
- **Валидация топологии (`validate_graph`):** при загрузке через `nx.descendants` проверяется, что все узлы достижимы из `startEvent` и ни один нетерминальный узел не является тупиком.
- **Извлечение ветки симптома (`get_symptom_branch`):** обход в глубину (`dfs`) от `startEvent` отсекает чужие ветки на шлюзе `Gateway_SymptomBranch` и собирает шаги регламента, правила эскалации и список обязательных инструментов. Отслеживание флага `in_conditional_escalation` гарантирует, что тулы, стоящие *за* условным шлюзом (например, `find_affected_products_by_lot` после проверки `lot_defect_rate > 0.10`), попадают в правила эскалации, но не навязываются как обязательные для нормальных партий.

### 3.2. Гибридный поиск по SOP (`../src/rag/indexer.py` + `../src/rag/hybrid_retriever.py`)
- **Индексация Markdown-инструкций:** `indexer.py` сканирует только `knowledge_base/sops/*.md`, режет документы по заголовкам `##` и `###` с сохранением названия документа в каждом чанке, кодирует их локальной CPU-моделью `cointegrated/LaBSE-en-ru` (768 dim, cosine) и загружает в Qdrant с детерминированными UUID (`uuid5`).
- **Двухконтурный поиск и RRF:** `HybridRetriever` параллельно запускает:
  1. **Лексический `BM25Okapi`** поверх `tokenize_bilingual`, который индексирует составные артикулы (`STAND-03`, `RS550-24V`, `JK-BMS-6S`) одновременно как целый токен и как разбитые подтокены.
  2. **Семантический поиск в Qdrant** по эмбеддингам запроса.
  Ранги обоих списков объединяются через **Reciprocal Rank Fusion (RRF)** с константой `rrf_k = 60` и настраиваемыми весами `rag_dense_weight` / `rag_bm25_weight`.

---

## 4. Чат-агент расследования (`LangGraph` / `../src/agent`)

### 4.1. Состояние графа (`AgentState` в `../src/agent/state.py`)
`AgentState` построен как `TypedDict` с двумя режимами обновления полей:
- **Перезаписываемые поля:** `intake`, `bpmn_guidance`, `investigator_draft`, `audit_result`, `final_verdict`, `action_result`, `tool_steps`, `revision_count`.
- **Накапливаемые поля (редьюсеры):** `messages` использует `add_messages`, а `evidence_log` и `token_usage` используют `operator.add` — за счёт этого сырые ответы SQL-тулов и биллинг токенов по каждому вызову никогда не затираются на циклах и повторных ревизиях.

### 4.2. Слой LLM и отказоустойчивый Structured Output (`../src/agent/llm.py`)
- **Разделение ролей (`AgentModelBundle`):** создаются 4 независимых клиента `ChatOpenAI` (через OpenRouter с `temperature=0.0`) под роли `FAST`, `INVESTIGATOR`, `AUDITOR` и `JUDGE`.
- **Защита `invoke_structured_with_usage`:**
  - адаптация протокола под семейство модели: для `deepseek` первой идёт попытка через `method="function_calling"`, для остальных — дефолтный `json_schema` с автоматическим ретраем на альтернативный метод при сбое;
  - жёсткий wall-clock таймаут `asyncio.wait_for(..., timeout=85.0)` для обрыва зависших HTTP keepalive-соединений;
  - трёхуровневый фолбэк в `_extract_parsed_and_usage`: если LangChain вернул `parsed=None`, код срезает markdown-обёртки из `raw.content` через `_clean_json_text` и валидирует JSON повторно, а при пустом тексте забирает словарь напрямую из `raw.tool_calls[0]["args"]`.
- **Тестовый двойник (`ScriptedChatModel`):** наследник `BaseChatModel`, реализующий `bind_tools`, `with_structured_output` и генерацию `usage_metadata` для детерминированных тестов и `--mode mock`.

### 4.3. Топология `StateGraph` и маршрутизация (`../src/agent/graph.py`)

```text
[START] ──> intake_node (FAST) ──(нет серийника)──> [END / уточнение]
                 │
            (серийник OK)
                 ▼
            triage_node (BPMN + Hybrid RAG, 0 LLM calls)
                 │
                 ▼
         investigator_node (INVESTIGATOR) <──┐
            │        ▲                       │ (feedback при REJECTED,
       (tool_calls)  │                       │  revision_count < 2)
            ▼        │                       │
         ToolNode ───┘                       │
            │ (draft готов)                  │
            ▼                                │
          auditor_node (AUDITOR: другое семейство!) ──┘
            │
            ├──(NEED_WEB_RESEARCH: БД 100% чиста)──> web_researcher_subgraph (FAST + DDGS)
            │                                              │
            └──(APPROVED / max_revisions)──────────────────┴──> verdict_node (INVESTIGATOR)
                                                                      │
                                                                      ▼
                                                                 action_node (HITL interrupt())
                                                                      │
                                                                 [END / PostgreSQL ACID commit]
```

Пошаговая работа узлов:
1. **`intake_node` (`FAST`):** извлекает серийник машинки, категорию симптома и условия эксплуатации. Если серийник не указан (`route_after_intake`), граф завершается с уточняющим вопросом и продолжает диалог в том же `thread_id` на следующем ходу.
2. **`triage_node` (0 LLM-вызовов):** детерминированно извлекает ветку BPMN и подтягивает релевантные SOP-правила через `HybridRetriever`.
3. **`investigator_node` ↔ `tools_node` (`INVESTIGATOR`):** ReAct-цикл поверх `ToolNode` и 7 типизированных read-only инструментов (`../src/agent/tools.py`: `get_item_genealogy`, `get_item_history`, `get_lot_quality_stats`, `find_affected_products_by_lot`, `get_instrument_channel_anomalies`, `get_operator_anomalies`, `search_sop_knowledge`). `tools_node` параллельно складывает каждый сырой JSON-ответ в `evidence_log`. После завершения проверок Следователь формирует структурированный `InvestigationDraft`.
4. **`auditor_node` (`AUDITOR` — другое семейство моделей) + кодовые guardrails:** Ревизор не видит внутренний монолог Следователя — он сверяет только `InvestigationDraft`, правила ветки BPMN и сырой `evidence_log`. При пропусках или расхождении цифр возвращает `REJECTED` с `feedback` (петля Reflection, до `max_revisions = 2`). Поверх LLM в коде зашиты три жёстких инварианта:
   - если `get_item_genealogy` вернул `found=False`, кейс принудительно закрывается со статусом `INCONCLUSIVE` без вызова веб-поиска;
   - переход в `NEED_WEB_RESEARCH` блокируется (`REJECTED`), пока не вызваны все 5 обязательных SQL-проверок по заводу;
   - если БД признана чистой (`db_is_clean=True` и `db_anomalies_found=False`), ответ `APPROVED` автоматически приводится к `NEED_WEB_RESEARCH`.
5. **`web_researcher_node` (`../src/agent/web_research.py`):** отдельный вложенный `StateGraph(WebResearchState)` из трёх шагов (`plan_queries → fetch_web → synthesize_hypotheses`) поверх DuckDuckGo (`ddgs`). Сырые сниппеты остаются внутри подграфа и не попадают в `messages` родительского графа — наружу выходят только 2–3 компактные гипотезы `ExternalHypothesis`.
6. **`verdict_node` (`INVESTIGATOR`):** формирует `FinalVerdict`, причём поле `external_hypotheses` жёстко перезаписывается в коде из выхода подграфа, чтобы модель не могла смешать внешние гипотезы с подтверждёнными фактами БД (`confirmed_db_facts`).
7. **`action_node` (HITL `interrupt()` + PostgreSQL ACID):** граф скомпилирован с чекпоинтером `AsyncPostgresSaver` (`psycopg_pool`, `JsonPlusSerializer` с разрешёнными Pydantic-моделями стейта). В `action_node` вызывается `interrupt(hitl_payload)` без `try/except` и без предварительной записи в БД; после получения `Command(resume={"approved": True, ...})` узел перезапускается, подставляет решение человека и в одной транзакции `async with session.begin():` создаёт записи в `support_tickets` и `recall_notices`.