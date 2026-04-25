# SlmFlowFramework

SlmFlow — фреймворк для быстрого прототипирования экспериментов SLM vs LLM в контексте RAG-пайплайнов. Предоставляет функционал для генерации синтетических данных, динамической маршрутизации SLM/LLM на этапе инференса и оценки качества генерации.

## Quick Start

**1. Клонировать репозиторий**
```bash
git clone https://github.com/MikhaelProskurin/itmo_slm_flow.git
cd itmo_slm_flow
```

**2. Создать виртуальное окружение и установить зависимости**
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

**3. Заполнить `.env`** на основе [образца ниже](#образец-env)

**4. (Опционально) Запустить локальный vLLM-сервер**
```bash
docker-compose up -d
```

## Зависимости

**Python:** 3.11+

- `langchain`, `langchain-openai` — LLM-интеграция
- `pydantic` — типизированные модели данных
- `aiofiles`, `asyncio` — асинхронный I/O
- `spacy` + `en_core_web_lg` — NLP-признаки
- `tiktoken` — подсчёт токенов
- `wordfreq` — частота слов
- `pandas` — табличные данные
- `bert-score`, `rouge` — метрики качества генерации
- `torch`, `transformers` — бэкенд BERTScore

## Образец .env

Фреймворк использует [OpenRouter](https://openrouter.ai/) как единую точку доступа к моделям. Имена моделей задаются непосредственно в ноутбуках.

```env
OPENROUTER_API_KEY=<your_openrouter_key>
BASE_URL=https://openrouter.ai/api/v1
```

## Архитектура

```
core/
├── messaging/
│   ├── builder.py      # LangchainMessageBuilder: реестр промптов и PydanticOutputParser
│   └── prompts.py      # PROMPT_REGISTRY, TASK_DESCRIPTIONS, все шаблоны промптов
├── data/
│   ├── synthetic.py    # RAGDatasetAsyncGenerator, DatasetDeclaration, Pydantic-модели генерации
│   └── datasets.py     # RAGSyntheticDataset, DatasetRecord, BaseDataset ABC
├── tasks/
│   └── rag.py          # RAGTask: единица инференса (запрос + документы → предсказание)
├── pipeline/
│   └── runner.py       # RAGPipelineRunner, InferenceRecord, EvaluationRecord, JScore
├── router/
│   ├── features.py     # RAGFeatureExtractor, RerankingVector, CompressionVector
│   ├── language_model_router.py  # LMRouter: маршрутизация RAGTask к SLM или LLM
│   └── policies.py     # WeightedRuleBasedRoutingPolicy, SLMRoutingPolicy, WeightedRule
└── utils/
    ├── additional_metrics.py   # compute_slm_routing_metrics, SLMRoutingMetrics
    └── representation.py       # Утилиты отображения Pandas
```

### Ключевые компоненты

**`messaging/builder.py`** — `LangchainMessageBuilder`: хранит реестр именованных шаблонов и соответствующих им `PydanticOutputParser`. `from_sequence()` — стандартная фабрика; `create_message()` рендерит `SystemMessage`, внедряя kwargs и инструкции форматирования; `get_parser()` возвращает парсер для обработки структурированного вывода.

**`messaging/prompts.py`** — все строки шаблонов промптов (`RERANKING_DATA_GENERATION`, `CONTEXT_COMPRESSION_DATA_GENERATION`, `RERANKING_INFERENCE`, `CONTEXT_COMPRESSION_INFERENCE`, `JUDGE_EVALUATION`), словарь `TASK_DESCRIPTIONS` и `PROMPT_REGISTRY` — замороженный (`frozen=True`) датакласс для именованного доступа к промптам.

**`data/synthetic.py`** — Pydantic-модели вывода генерации (`RAGDocument`, `RerankingSample`, `CompressionSample`, `PersistentSample`) и `DatasetDeclaration` (задачи/домены/сложности/размер батча). `RAGDatasetAsyncGenerator` управляет асинхронным циклом генерации: перебирает все комбинации task×domain×difficulty, параллельно вызывает LLM через семафор, парсит структурированный вывод и сохраняет каждый пример как UUID-именованный JSON.

**`data/datasets.py`** — `StandardSample` и `DatasetRecord` (единый формат строки в памяти), `BaseDataset` (ABC с `from_files`, `to_pandas`, `__len__`, `__getitem__`) и `RAGSyntheticDataset`: рекурсивно загружает JSON из `{root}/{task}/{domain}/{difficulty}/{uuid}.json`, определяет метаданные из компонентов пути, перемешивает при загрузке.

**`tasks/rag.py`** — `RAGTask`: обёртка единицы инференса (name, query, documents). `from_record()` конструирует из `DatasetRecord`; `agenerate_prediction()` рендерит промпт через `LangchainMessageBuilder`, асинхронно вызывает модель, парсит через `PydanticOutputParser` и возвращает `"structured_output_parsing_error"` при `OutputParserException`.

**`pipeline/runner.py`** — `RAGPipelineRunner`: управляет тремя клиентами `ChatOpenAI` (SLM, LLM, judge) и `LMRouter`. `arun()` итерирует датасет, маршрутизирует каждую строку, конкурентно собирает предсказания и возвращает `InferenceRecord`. `aevaluate()` оценивает записи с помощью BERTScore, ROUGE и LLM-судьи, возвращая `EvaluationRecord`. Определяет `JScore`, `InferenceRecord`, `EvaluationRecord`, `RerankingMetrics`, `CompressionMetrics`.

**`router/features.py`** — `RAGFeatureExtractor`: использует spaCy (noun chunks, леммы, косинусное сходство) и tiktoken (подсчёт токенов) для вычисления векторов признаков. Диспетчеризует к `compute_reranking_feature_vector()` или `compute_compression_feature_vector()` по имени задачи. Определяет `RAGFeatureVectorBase`, `RerankingVector`, `CompressionVector`.

**`router/language_model_router.py`** — `LMRouter`: сопоставляет режим маршрутизации (`"slm"`, `"llm"`, `"dynamic"`) с выбором модели. `select_language_model()` извлекает признаки и возвращает `(fvector, route)`, где route — `"_slm"` или `"_llm"`.

**`router/policies.py`** — `Routable` (структурный протокол с `call_large_model()`), `WeightedRule` (именованное пороговое правило с весом), `WeightedRuleBasedRoutingPolicy` (запускает LLM-маршрутизацию при выполнении порогов по числу и суммарному весу правил) и `SLMRoutingPolicy` (делегирует решение о маршрутизации SLM-клиенту).

**`utils/additional_metrics.py`** — `SLMRoutingMetrics` и `compute_slm_routing_metrics()`: вычисляют эффективность SLM-маршрутизации (`slm_success_ratio`, `slm_routing_ratio`) из списка `EvaluationRecord` по заданному порогу jscore.

## Режимы маршрутизации

| Режим | Описание |
|-------|----------|
| `"slm"` | Всегда использует малую модель |
| `"llm"` | Всегда использует большую модель |
| `"dynamic"` | Динамический выбор: диспетчеризует к `WeightedRuleBasedRoutingPolicy` или `SLMRoutingPolicy` в зависимости от типа переданной политики |

## Датасет

`slm_flow_df/` — основной датасет синтетических RAG-примеров:

| Измерение | Значения |
|-----------|----------|
| Задача | `reranking`, `context_compression` |
| Домен | `coding`, `history`, `math`, `medicine`, `research` |
| Сложность | `easy`, `medium`, `complex` |

Структура файлов: `slm_flow_df/{task}/{domain}/{difficulty}/{uuid}.json`

Каждый файл содержит `query`, `documents` (список `idx` + `content`) и `golden_answer`.