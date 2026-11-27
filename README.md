# LangVeyor — Мультиагентная MES RCA-система на LangGraph

Production-grade портфолио-проект: **мультиагентная система расследования первопричин неисправностей (Root Cause Analysis, RCA)** поверх конвейерной MES-архитектуры (Домены 0–4 + новый Домен 5 `DB-05` + алгоритм стратифицированной выборки `ALG-001`).

Предметная область — сборочное производство **прокатных детских электромашинок для парков аттракционов** (`TM-BAT-6S`, `TM-BODY`, `TM-CAR-FINAL`).

---

## Архитектура системы

```text
[Оператор CLI / Обращение из парка]
               │
               ▼
       ┌──────────────┐   нет серийника    ┌────────────────────────┐
       │ intake_node  │───────────────────>│ Уточнение серийника    │
       │ (FAST LLM)   │                    └────────────────────────┘
       └──────┬───────┘
              │ серийник OK
              ▼
       ┌──────────────┐   knowledge_base/support_process.bpmn (NetworkX)
       │ triage_node  │<─ knowledge_base/sops/*.md (Qdrant LaBSE-en-ru + BM25 RRF)
       │ (0 LLM calls)│
       └──────┬───────┘
              ▼
    ┌───────────────────┐     tool_calls     ┌──────────────────┐
    │ investigator_node │<──────────────────>│    tools_node    │
    │ (INVESTIGATOR)    │   evidence_log     │ 7 typed SQL/RAG  │
    └─────────┬─────────┘   (PostgreSQL 17)  │     тулов        │
              │ InvestigationDraft           └──────────────────┘
              ▼
       ┌──────────────┐   REJECTED (rev < 2)
       │ auditor_node │──────────────────────> обратно в investigator_node
       │ (AUDITOR:    │
       │  другое сем.)│   NEED_WEB_RESEARCH  ┌──────────────────────────┐
       └──────┬───────┴─────────────────────>│ web_researcher_node      │
              │ APPROVED                     │ (Isolated Subgraph+DDGS) │
              ▼                              └────────────┬─────────────┘
       ┌──────────────┐                                   │
       │ verdict_node │<──────────────────────────────────┘
       │ (Факты БД != │  confirmed_db_facts vs external_hypotheses
       │  Веб-гипотез)│
       └──────┬───────┘
              ▼
       ┌──────────────┐   interrupt() -> Command(resume=...)
       │ action_node  │──────────────────────────────────────> PostgreSQL ACID:
       │ (HITL пауза) │   AsyncPostgresSaver                   support_tickets /
       └──────────────┘                                        recall_notices
```

---

## Технологический стек

- **Python 3.12+**, управление зависимостями через **`uv`** (`pyproject.toml` + `uv.lock`).
- **LangGraph** (`StateGraph`, `ToolNode`, `interrupt()` / `Command(resume=...)`, `AsyncPostgresSaver` на `psycopg_pool`).
- **LLM через OpenRouter** (`ChatOpenAI` с 4 независимыми ролями: `OPENROUTER_MODEL_FAST`, `OPENROUTER_MODEL_INVESTIGATOR`, `OPENROUTER_MODEL_AUDITOR` и `OPENROUTER_MODEL_JUDGE` из разных семейств моделей).
- **PostgreSQL 17** + SQLAlchemy 2.1 Async + Alembic (Домены 0–4 + Домен 5 `DB-05`, рекурсивные `WITH RECURSIVE` CTE для генеалогии `ASSEMBLED_INTO -> SPLIT` и прямого Recall-анализа).
- **Qdrant + Hybrid RAG**: локальные CPU-эмбеддинги `cointegrated/LaBSE-en-ru` (`sentence-transformers`) + локальный `BM25Okapi` с Reciprocal Rank Fusion (RRF).
- **BPMN 2.0 XML Parser**: `lxml` + `networkx.DiGraph` (без нарезки XML в векторную БД).
- **Веб-поиск:** DuckDuckGo (`ddgs`) без API-ключей, изолированный подграф с fallback.
- **CLI:** интерактивная async-консоль на `rich` с live-стримингом шагов, вызовов тулов и HITL-подтверждением.

---

## Быстрый старт (Quickstart)

1. **Подготовьте `.env`:**
   ```bash
   cp .env.example .env
   # Опционально впишите OPENROUTER_API_KEY для live-режима (без ключа CLI и evals работают в режиме --mock)
   ```
2. **Поднимите PostgreSQL 17 и Qdrant:**
   ```bash
   make up
   ```
3. **Установите зависимости и примените миграции:**
   ```bash
   uv sync
   make migrate
   ```
4. **Засейте данные за 1 год производства (~451 машинка, ~18 партий АКБ, 5 аномалий):**
   ```bash
   make seed
   ```
5. **Проиндексируйте 5 SOP-регламентов в Qdrant (`cointegrated/LaBSE-en-ru`):**
   ```bash
   make index
   ```
6. **Запустите интерактивную консоль или evals:**
   ```bash
   make cli
   # или демонстрационное меню:
   uv run python -m src.cli.chat --demo
   # прогон тестов и evals:
   make verify-all
   ```

---

## Команды `Makefile`

| Цель | Описание |
| :--- | :--- |
| `make up` | Поднять PostgreSQL 17 и Qdrant в Docker Compose и дождаться `healthy` |
| `make down` | Остановить контейнеры |
| `make migrate` | Применить миграции Alembic (`alembic upgrade head`) |
| `make seed` | Запустить детерминированный сидер на 12 месяцев (`src.db.seed`) |
| `make index` | Проиндексировать SOP в Qdrant (`src.rag.indexer`) |
| `make cli` | Запустить интерактивную консоль `rich` (`src.cli.chat`) |
| `make lint` | Проверить код линтером и форматтером `ruff` |
| `make typecheck` | Статическая проверка типов `mypy src tests evals` |
| `make test` | Запустить все тесты `pytest -v` (включая реальный Postgres, Qdrant и HITL) |
| `make eval-mock` | Прогнать Golden Dataset в детерминированном режиме `--mode mock` |
| `make eval-live` | Прогнать Golden Dataset через реальный OpenRouter API (`--mode live`) |
| `make verify-all` | Полный зелёный конвейер: `up → migrate → seed → index → lint → typecheck → test → eval-mock` |

---

## Посаженные аномалии (Ground Truth для ручной проверки и Evals)

| Кейс / Серийник | Что посажено в БД | Что должна сделать система |
| :--- | :--- | :--- |
| **`CAR-2026-0412`** | Партия `LOT-BAT-2026-014` (май 2026): **20% брака** по разрядной ёмкости (`< 33.0 А·ч`); 12 годных/перепроверенных АКБ ушли в готовые машинки | Раскрутить генеалогию `ASSEMBLED_INTO → SPLIT`, увидеть `> 10%` брака, вызвать `find_affected_products_by_lot` и вывести **все 12 машинок** под Recall |
| **`CAR-2026-0388`** | АКБ заряжена на **`STAND-03`, канал 7** в октябре 2026 (поверка стенда истекла `2026-09-30`, систематический недозаряд на канале 7 vs остальные каналы) | Подтвердить дрейф канала 7 и просрочку поверки, предложить `CALIBRATION_HOLD` |
| **`CAR-2026-0295`** | Сборка мотора `RS550-24V` оператором **Петров А.В.** (`OP-02`) в августе 2026 с всплеском `REWORK` по перетяжке момента (`> 12.0 Н·м`) | Выявить аномалию оператора в окне ±14 дней против базовой линии цеха |
| **`CAR-2026-0199`** | В истории есть `REWORK` на тестовом заезде и **закрытый тикет** `support_tickets` с тем же симптомом | Включить ветку BPMN `Gateway_RepeatTicket` и эскалировать (`ESCALATED`, `CRITICAL`) |
| **`CAR-2026-0777`** | **100% чистая БД**: 0 `REWORK`, 0% брака партии `LOT-BAT-2026-015`, свежие поверки, лучший оператор. BOM: `RS550-24V`, `JK-BMS-6S` | Признать БД чистой (`NEED_WEB_RESEARCH`), запустить изолированный `web_researcher_node` и отделить веб-гипотезы от фактов БД |

---

## Дополнительная документация

- [reference_docs/ARCHITECTURE.md](file:///home/iron-felix/_projects/LangVeyor/reference_docs/ARCHITECTURE.md) — сводная архитектура Доменов 0–4 + Домен 5 (`DB-05`), `ALG-001` и LangGraph.
- [docs/EVALS_INTERVIEW_GUIDE.md](file:///home/iron-felix/_projects/LangVeyor/docs/EVALS_INTERVIEW_GUIDE.md) — подробный инженерный гайд по Evals мультиагентных систем для подготовки к собеседованиям.
- [evals/last_report.md](file:///home/iron-felix/_projects/LangVeyor/evals/last_report.md) — последний сгенерированный отчёт по метрикам Evals.
