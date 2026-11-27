# ARCHITECTURE: Мультиагентная MES RCA-система (Домены 0–4 + Домен 5 `DB-05`)

> **Статус:** Утверждённая архитектура проекта LangVeyor  
> **Базовая платформа:** SW005 MES (Домены 0–4: [`DB-00`](./database/DB-00_master_schema.md), [`DB-01`](./database/DB-01_user_and_role_administration.md), [`DB-02`](./database/DB-02_user_authentication_and_session_management.md), [`DB-03`](./database/DB-03_technical_chart_and_conveyor_configuration.md), [`DB-04`](./database/DB-04_instruction_management_and_acknowledgement.md), [`ARCH-00`](./backend/ARCH-00_backend_specification.md))  
> **Новое расширение:** Домен 5 (`DB-05` — Исполнение маршрутных листов, генеалогия изделий, алгоритм `ALG-001`, сервисные тикеты/отзывы) + мультиагентная система расследования первопричин (RCA) на LangGraph + гибридный RAG (BPMN 2.0 + Qdrant + BM25)

---

## 1. Переиспользование принципов Доменов 0–4

Все базовые архитектурные стандарты эталонной документации SW005 сохраняются без изменений:

1. **Единые стандарты PostgreSQL 17 (`DB-00`, `ARCH-00 §5.2`):**
   - Строки перечисления хранятся как `VARCHAR(...)` с `CHECK`-ограничениями и зеркалируются в Python через `StrEnum` (никаких DDL `CREATE TYPE ...`).
   - Все временные метки — `TIMESTAMPTZ` с `clock_timestamp()` и `UtcDateTime` TypeDecorator.
   - Первичные ключи — `UUID` (`gen_random_uuid()`), за исключением высокочастотного журнала замеров `step_measurements` (`BIGINT GENERATED ALWAYS AS IDENTITY`).
   - Единая конвенция именования ограничений `POSTGRES_NAMING_CONVENTION` (`pk_`, `uq_`, `fk_`, `idx_`, `chk_`) и управление схемой строго через Alembic.
2. **Кадровый контур и RBAC (`DB-01`):**
   - Переиспользованы `roles` и `users` (строго одна роль на сотрудника, разделение ФИО, `is_active`).
   - Отказ от жёсткого удаления производственного персонала (`ON DELETE RESTRICT` в журналах исполнения).
3. **Конфигуратор конвейера и техкарт (`DB-03`):**
   - Переиспользованы `workstations`, `operations`, `tech_maps`, `tech_map_nodes`, `tech_map_node_guards`, `tech_map_sampling_rules`, `tech_map_transitions`.
   - Сохранены правило «двух ключей» (`is_approved` Технолога + `is_active` Начальника участка, `approved_by_id != activated_by_id`), иммутабельность утверждённых карт (Copy-on-Write), валидация DAG по алгоритму Кана и составные внешние ключи (`(tech_map_id, from_node_id)`, `(node_id, measurement_guard_id)`), исключающие кросс-картовые и кросс-узловые привязки.
4. **Оборудование и метрологическая прослеживаемость:**
   - Введена таблица `instruments` (зарядные стенды №1–3 по 16 каналов, измерители внутреннего сопротивления, динамометрические ключи, испытательные стенды) с обязательной датой следующей поверки `calibration_due_at`, привязываемая к фактам исполнения `step_executions`.
5. **Нормативная документация (`DB-04` -> `knowledge_base/`):**
   - Вместо S3-хранилища бинарных файлов в рамках RCA-агента регламенты вынесены в версионируемую базу знаний: структурированный процесс обработки обращений **BPMN 2.0 XML** (`knowledge_base/support_process.bpmn`) и 5 инженерных **SOP** (`knowledge_base/sops/*.md`), индексируемых локальной моделью `cointegrated/LaBSE-en-ru` в Qdrant + BM25 RRF.

---

## 2. Расширения статики (`DB-03` v2) и Домен 5 (`DB-05`)

### 2.1. Расширения таблиц Домена 3 (`DB-03`)

| Таблица | Новые поля | Назначение |
| :--- | :--- | :--- |
| `tech_maps` | `stop_min_defects INT`, `stop_defect_slope NUMERIC(6,4)` | Параметры $d_{min}$ и $R_{slope}$ линейного стоп-крана партии из алгоритма `ALG-001` |
| `tech_map_nodes` | `execution_scope VARCHAR(20)` (`'LOT' \| 'ITEM'`) | Область действия узла конвейера (партионная или поштучная) живёт на узле, а не на отдельном гарде |
| `tech_map_sampling_rules` | `escalation_sample_size INT`, `max_escalations INT` | Параметры динамического добора выборки при срабатывании дефекта |

### 2.2. Производственные техкарты предметной области (Сборка детских электромашинок)

1. **`TM-BAT-6S`** (*Приёмка и зарядка АКБ 6S 35Ah*): гибридный маршрут `LOT → SPLIT → ITEM`
   - Узел 1 (`LOT`): входной контроль партии и тары (`OP-LOT-RECEIVE`, `WS-BAT-IN`).
   - Узел 2 (`SPLIT`): распаковка и расщепление корневой партии на индивидуальные серийные маршрутные листы АКБ (`OP-LOT-SPLIT`, `WS-BAT-SPLIT`, создаёт связи `route_sheet_links.link_type = 'SPLIT'`).
   - Узел 3 (`ITEM`): визуальный осмотр корпуса/клемм (`OP-BAT-VISUAL`).
   - Узел 4 (`ITEM`): зарядка и балансировка на многоканальном стенде (`STAND-01..03`, каналы `1..16`, `OP-BAT-CHARGE`).
   - Узел 5 (`ITEM`): электрические испытания (напряжение $21.6\text{--}25.2\,\text{В}$, ёмкость $33.0\text{--}37.5\,\text{А}\cdot\text{ч}$, внутреннее сопротивление $8.0\text{--}18.0\,\text{мОм}$) с петлёй `REWORK`, линейным стоп-краном и стратифицированной выборкой `ALG-001`.
2. **`TM-BODY`** (*Сборка кузова машинки*): поштучный маршрут `ITEM`
   - Контроль покраски и пластика → сборка рулевого узла и сиденья → проверка крепежа и жгутов (с петлёй `REWORK`).
3. **`TM-CAR-FINAL`** (*Монтаж ходовой части с электромотором и финальная сборка*): поштучный маршрут `ITEM`
   - Установка редуктора и DC-мотора `RS550-24V`, контроль момента затяжки (`8.0–12.0 Н·м`) → узел комплектации/сборки (создаёт связи `ASSEMBLED_INTO` из готовых `TM-BODY` + `TM-BAT-6S` + контроллера `JK-BMS-6S`) → тестовый заезд → финальный ОТК.

### 2.3. Схема ядра исполнения и генеалогии (`DB-05`)

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

Ключевые таблицы Домена 5:
1. **`route_sheets`** — единый курсор конечного автомата как для партий (`LOT`), так и для штучных изделий (`ITEM`). Поле `hold_reason` (nullable) отделено от рабочего `status` (`IN_PROGRESS`, `SPLIT_OPEN`, `SPLIT_CLOSED`, `COMPLETED`, `DEFECT`, `QUARANTINE`), чтобы блокировка партии не затирала технологический статус. Денормализованная индексированная ссылка `root_lot_id` обеспечивает $O(1)$ проверку блокировки корневой партии и доступ к счётчикам `defect_count` / `processed_count` для `ALG-001`.
2. **`lot_metadata`** — пассивная `1:1` анкета корневых партий (`supplier_name`, `invoice_number`, `arrival_date`, `component_specs JSONB` со спецификацией BOM: модель мотора, BMS, химии ячеек).
3. **`route_sheet_links`** — направленный граф материальной генеалогии `source_rs_id -> target_rs_id` (`SPLIT` для $1 \to N$ расщепления партии; `ASSEMBLED_INTO` для $N \to 1$ сборки машинки). Составные индексы `(source_rs_id, link_type)` и `(target_rs_id, link_type)` обеспечивают быстрый двунаправленный обход через `WITH RECURSIVE`:
   - **Назад (Genealogy):** от серийника машинки к кузову, АКБ, корневой партии АКБ и её BOM/поставщику;
   - **Вперёд (Recall Analysis):** от проблемной партии ко всем собранным готовым машинкам.
4. **`step_executions`** — append-only журнал фактов прохождения операций (`route_sheet_id`, `node_id`, `operator_id`, `instrument_id`, `channel_no`, `group_batch_code`, `outcome`, `applied_transition_id`, `started_at`, `completed_at`).
5. **`step_measurements`** — append-only журнал физических/логических замеров (`BIGINT IDENTITY`, `execution_id`, `guard_id`, `numeric_value`, `boolean_value`, `text_value`, `is_passed`, `measured_at`).
6. **`sampling_node_states`** — персистентное состояние стратифицированного окна `ALG-001` с ключом `(lot_rs_id, node_id)`: `m_rem`, `n_rem`, `w_rem`, `taken_in_window`, `escalations_used`. Обновляется транзакционно под блокировкой `SELECT ... FOR UPDATE`.
7. **`support_tickets`** и **`recall_notices`** — журналы обращений и отзывов, пополняемые через подтверждённые оператором действия (HITL `interrupt()` в `action_node`).

---

## 3. Математическая модель алгоритма `ALG-001` (`src/mes/sampling.py`)

1. **Детерминированный PRNG на базе `HMAC-SHA256`:**
   Для пары `(lot_rs_id, node_id, item_tracking_code)` вычисляется 64-битное равномерное значение $u \in [0, 1)$:
   $$u = \frac{\text{int}(\text{HMAC\_SHA256}(\text{key}, \text{lot\_rs\_id}:\text{node\_id}:\text{item\_code})[:8])}{2^{64}}$$
   Это исключает угадывание выборки оператором и гарантирует 100% воспроизводимость при аудите.
2. **Стратифицированный метод Фана–Мюллера–Резухи (Algorithm S):**
   При инициализации партии объёмом $N$ с целевой выборкой $m$ формируется окно $w_{rem} = \max(1, \lfloor n_{rem} / m_{rem} \rfloor)$ при $n_{rem} > m_{rem}$. Внутри окна с `taken_in_window == 0` элемент отбирается с вероятностью $p = 1 / w_{rem}$ (гарантия ровно 1 выборки в каждой страте). После отбора `taken_in_window` инкрементируется, `m_rem` уменьшается на 1, и остаток окна пропускается.
3. **Триада состояний добора (`n_rem` vs `m_rem`):**
   При обнаружении дефекта в выборке (до `max_escalations` раз) к `m_rem` добавляется `escalation_sample_size`:
   - `n_rem < m_rem`: дефицит образцов — включается 100% сплошной контроль остатка ($p = 1.0$) с фиксацией нехватки `shortfall = m_rem - n_rem`.
   - `n_rem == m_rem`: точное совпадение — все оставшиеся изделия отбираются детерминированно ($p = 1.0$).
   - `n_rem > m_rem`: пересчитывается уплотнённое стратифицированное окно $w_{rem} = \max(1, \lfloor n_{rem} / m_{rem} \rfloor)$ и продолжается равномерный отбор.
4. **Корректировка `delta` при недопоставке:**
   При расхождении фактического количества `n_rem <- max(0, n_rem - delta)` окно пересчитывается с сохранением инварианта достаточности: если `taken_in_window >= 1`, остаток окна ограничивается $w_{rem} \le n_{rem} - m_{rem}$; если $n_{rem} \le m_{rem}$, система немедленно переходит в режим 100% контроля.
5. **Линейный стоп-кран партии:**
   Партия переводится в блокировку `hold_reason = 'DEFECT_RATE_EXCEEDED'` (без затирания рабочего `status`), если:
   $$\text{defect\_count} \ge d_{min} + R_{slope} \times \text{processed\_count}$$
   Это отсекает ложные срабатывания на первых 1–2 единицах и останавливает конвейер при статистически устойчивом всплеске брака.

---

## 4. Мультиагентная архитектура RCA на LangGraph и Гибридный RAG

### 4.1. Разделение знаний: BPMN 2.0 vs SOP RAG
* **Процессный граф (`knowledge_base/support_process.bpmn`):** НЕ нарезается в векторную БД. Парсится через `lxml` в `networkx.DiGraph` (`src/rag/bpmn_parser.py`). По категории симптома извлекается детерминированная ветка шагов, обязательных проверок, шлюзов и порогов эскалации (например, порог брака партии $> 10\% \implies$ обязательный Recall).
* **Инженерные SOP (`knowledge_base/sops/*.md`):** нарезаются по заголовкам Markdown и индексируются в **Qdrant** с локальными CPU-эмбеддингами **`cointegrated/LaBSE-en-ru`** (768 dim, cosine) + локальный **BM25** (`rank_bm25`). Слияние выполняется через **Reciprocal Rank Fusion (RRF)**.

### 4.2. Топология `StateGraph` (7 узлов + `AsyncPostgresSaver`)

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

**Ключевые гарантии надёжности:**
1. **Context Hygiene:** `web_researcher_node` реализован как изолированный Subgraph — сырой веб-текст DuckDuckGo фильтруется внутри него и в главный `AgentState` попадают только 2–3 компактные структурированные `ExternalHypothesis`.
2. **Независимая ревизия (Cross-Family Reflection):** `auditor_node` использует модель другого семейства (`OPENROUTER_MODEL_AUDITOR`) и сверяет черновик строго против сырого `evidence_log` SQL-тулов, пресекая галлюцинации цифр и пропущенные по BPMN шаги.
3. **Разделение правды и гипотез:** `confirmed_db_facts` и `external_hypotheses` хранятся в раздельных полях состояния и отчёта.
4. **Строгий HITL `interrupt()`:** в `action_node` нет `try/except` вокруг `interrupt()` и нет записей в БД до подтверждения; при `Command(resume={"approved": True, ...})` создание `support_tickets` и `recall_notices` выполняется в одной ACID-транзакции PostgreSQL.
