# SlmFlowFramework

SlmFlow - это фреймвокр для быстрого прототипирования экспериментов SLM vs LLM в контексте RAG пайплайнов. Данный фреймворк предоставляет функционал для работы с данными (генерации и интеграции существующих датасетов), динамической маршрутизации SLM vs LLM на этапе инференса, а также инструменты оценки качества генерации.

## Quick Start

**1. Клонировать репозиторий**
```bash
git clone https://github.com/MikhaelProskurin/itmo_slm_flow.git
cd <your_clone_directory>
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

**5. Изучить пример** — [example.ipynb](example.ipynb)

## Зависимости

**Python:** 3.11+

- `langchain`
- `pydantic`
- `asyncio`, `aiofiles`
- `spacy`
- `tiktoken`
- `wordfreq`
- `pandas`
- `scikit-learn`
- `PyTorch`

## Образец .env

```env
# keys
OPENAI_API_KEY=<your_openai_key>
VLLM_API_KEY=<your_vllm_key>
OPENROUTER_API_KEY=<your_openrouter_key>

# model names
LLM_MODEL_NAME=<llm_model_name>
SLM_MODEL_NAME=<slm_model_name>
JUDGE_MODEL_NAME=<judge_model_name>
GENERATOR_MODEL_NAME=<generator_model_name>
```

## Архитектура

```
core/
├── io/
│   ├── models.py       # Pydantic-модели данных (входные/выходные форматы)
│   └── prompts.py      # Реестр промптов (генерация, инференс, оценка)
├── data/
│   ├── synthetic.py    # Асинхронный генератор синтетических RAG-примеров
│   └── datasets.py     # Загрузка датасетов, абстрактный интерфейс
├── tasks/
│   └── base.py         # Абстракция задачи и реализация RagTask
├── flow/
│   ├── base.py         # Оркестрация пайплайна инференса
│   └── utils.py        # LLM-судья, метрики (BERTScore, ROUGE), таймер
└── scheduler/
    ├── base.py         # Маршрутизатор SLM/LLM (llm_only / slm_only / dynamic)
    ├── features.py     # Извлечение признаков из задачи (spaCy + tiktoken)
    ├── policies.py     # Политики маршрутизации (пороговая и ML-классификатор)
    └── utils.py        # NLP-утилиты для вычисления признаков
```

### Ключевые компоненты

**`io/models.py`** — единый источник истины для всех Pydantic-моделей: форматы синтетических данных, строки датасета, векторы признаков, результаты задач и записи инференса.

**`io/prompts.py`** — все промпты для генерации, инференса и оценки определены как строки на уровне модуля и экспонируются через `PROMPT_REGISTRY` — неизменяемый (`frozen=True`) синглтон-датакласс. Обращаться к промптам следует только через него, не создавая экземпляры `PromptRegistry` напрямую.

**`data/synthetic.py`** — генерирует RAG-примеры через API LLM провайдера, сохраняет JSON-чекпоинты по пути `slm_flow_df/{task}/{domain}/{difficulty}/{uuid}.json`.

**`scheduler/features.py`** — Модуль экстракторов признаков, реализующих интерфейсы абстрактного класса `BaseFeatureExtractor` для RAG-speciefic задач. 

**`scheduler/policies.py`** — Модуль, содержащий политики маршрутизации, которые реализуют `BaseRoutingPolicy.decide()` и возвращают `True` для LLM, `False` для SLM.

**`flow/base.py`** — `InferenceFlow` итерирует датасет, вызывает планировщик, выполняет задачу и асинхронно оценивает результат через LLM-судью. Возвращает DataFrame с `FlowResult` по каждому примеру.

## Датасет

`slm_flow_df` — основной датасет синтетических RAG-примеров:

| Измерение  | Значения |
|------------|----------|
| Задача     | `reranking`, `context_compression` |
| Домен      | `coding`, `history`, `math`, `medicine`, `research` |
| Сложность  | `easy`, `medium`, `complex` |

Каждый файл содержит `query`, `documents` (список `idx` + `content`) и `golden_answer`, а также дополнительные метаданные.
