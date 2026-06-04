# SlmFlowFramework

Фреймворк для экспериментов с динамической маршрутизацией SLM/LLM в RAG-пайплайнах. Идея простая: не гонять дорогую большую модель на каждый запрос, а определять на лету — справится ли малая. Фреймворк закрывает весь цикл: генерация синтетических данных → маршрутизация на инференсе → оценка качества.

> Подробное описание архитектуры, стратегий маршрутизации и метрик — в [техническом отчёте](REPORT.md).

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

**3. Создать `.env`** со следующим содержимым:

```env
OPENROUTER_API_KEY=<your_openrouter_key>
BASE_URL=https://openrouter.ai/api/v1
```

Фреймворк использует [OpenRouter](https://openrouter.ai/) как единую точку доступа к моделям. Конкретные имена моделей задаются в ноутбуках.

## Структура проекта

```
core/
├── messaging/
│   ├── builder.py      # LangchainMessageBuilder: реестр промптов и PydanticOutputParser
│   └── prompts.py      # PROMPT_REGISTRY, TASK_DESCRIPTIONS, все шаблоны промптов
├── data/
│   ├── synthetic.py    # RAGDatasetAsyncGenerator, DatasetDeclaration, Pydantic-модели генерации
│   └── datasets.py     # RAGSyntheticDataset, RAGBenchDataset, DatasetRecord, BaseDataset ABC
├── tasks/
│   └── rag.py          # RAGTask: единица инференса (запрос + документы → предсказание)
├── pipeline/
│   └── runner.py       # RAGPipelineRunner, InferenceRecord, EvaluationRecord, JScore
├── router/
│   ├── features.py     # RAGFeatureExtractor, RerankingVector, CompressionVector, QuestionAnsweringVector
│   ├── language_model_router.py  # LMRouter: маршрутизация RAGTask к SLM или LLM
│   └── policies.py     # WeightedRuleBasedRoutingPolicy, SLMRoutingPolicy, Routable
└── utils/
    ├── additional_metrics.py   # compute_slm_routing_metrics, get_evaluation_summary
    └── representation.py       # утилиты отображения Pandas
```

## Режимы маршрутизации

| Режим | Поведение |
|-------|-----------|
| `"slm"` | Всегда малая модель — базовый бейзлайн |
| `"llm"` | Всегда большая модель — верхняя граница качества |
| `"dynamic"` | Решение принимается для каждого запроса: либо через правила по признаковому вектору (`WeightedRuleBasedRoutingPolicy`), либо через SLM-роутер (`SLMRoutingPolicy`) |

## Задачи и датасет

Синтетические данные хранятся в `slm_flow_df/` с организацией по трём осям:

| Измерение | Значения |
|-----------|----------|
| Задача | `reranking`, `context_compression`, `question_answering` |
| Домен | `coding`, `history`, `math`, `medicine`, `research` |
| Сложность | `easy`, `medium`, `complex` |

Структура файлов: `slm_flow_df/{task}/{domain}/{difficulty}/{uuid}.json`

Для задачи `question_answering` также поддерживается загрузка из внешних бенчмарков (RAGBench) через `RAGBenchDataset.from_files()`.
