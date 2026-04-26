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
- `pandas` — табличные данные

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

**`messaging/`** — слой промптов. `prompts.py` хранит все шаблоны и `PROMPT_REGISTRY`; `builder.py` (`LangchainMessageBuilder`) рендерит промпты и предоставляет `PydanticOutputParser` для структурированного разбора ответов модели.

**`data/synthetic.py`** — генерация датасета. `RAGDatasetAsyncGenerator` перебирает все комбинации task × domain × difficulty, конкурентно вызывает LLM и сохраняет результаты в JSON.

**`data/datasets.py`** — загрузка датасета. `RAGSyntheticDataset` рекурсивно читает JSON из `slm_flow_df/`, восстанавливает метаданные из структуры пути и отдаёт записи в виде `DatasetRecord`.

**`tasks/rag.py`** — единица инференса. `RAGTask` принимает запрос и документы, асинхронно вызывает модель и возвращает структурированный ответ.

**`pipeline/runner.py`** — оркестратор эксперимента. `RAGPipelineRunner` прогоняет датасет через маршрутизатор и модели (`arun()`), затем оценивает результаты BERTScore, ROUGE и LLM-судьёй (`aevaluate()`).

**`router/`** — маршрутизация SLM/LLM. `features.py` извлекает признаки запроса через spaCy и tiktoken; `language_model_router.py` (`LMRouter`) выбирает модель по режиму; `policies.py` реализует стратегии: `WeightedRuleBasedRoutingPolicy` (правила + веса) и `SLMRoutingPolicy` (решение делегируется SLM).

**`utils/additional_metrics.py`** — постобработка. `compute_slm_routing_metrics()` считает `slm_success_ratio` и `slm_routing_ratio` по заданному порогу jscore.

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