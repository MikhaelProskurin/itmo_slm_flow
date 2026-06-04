# SlmFlowFramework: динамическая маршрутизация SLM/LLM в RAG-пайплайнах

## О документе

Этот отчёт — детальное описание устройства SlmFlowFramework изнутри. Он предполагает, что общее представление о проекте уже есть (см. [README](README.md)), и сразу переходит к тому, как именно всё устроено.

Разделы документа:

- **Архитектура системы** — модульная структура и схема потока данных
- **Генерация синтетических данных** — асинхронный генератор, Pydantic-схемы выходов, структура файлов
- **Промпт-инжиниринг и структурированный вывод** — реестр промптов, паттерн LangchainMessageBuilder + PydanticOutputParser
- **Извлечение признаков** — числовые характеристики запроса и документов для принятия решения о маршрутизации
- **Политики маршрутизации** — детали `WeightedRuleBasedRoutingPolicy` и `SLMRoutingPolicy`, логика `LMRouter`
- **Пайплайн инференса и оценки** — устройство `arun()` и `aevaluate()`, метрики качества, агрегация результатов
- **Технические решения** — конкурентность, обработка ошибок парсинга, загрузка датасета

---

## Архитектура системы

### Общая схема потока данных

```
┌──────────────────────────────────────────────────────────────────┐
│              RAGSyntheticDataset / RAGBenchDataset               │
│  slm_flow_df/{task}/{domain}/{difficulty}/{uuid}.json            │
└─────────────────────────────┬────────────────────────────────────┘
                              │  DatasetRecord × N
                              ▼
                    ┌─────────────────┐
                    │    RAGTask      │  query + documents
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │          LMRouter            │
              │  ┌──────────────────────┐    │
              │  │  RAGFeatureExtractor │    │
              │  │  (spaCy + tiktoken)  │    │
              │  └──────────┬───────────┘    │
              │             │FeatureVector   │
              │             ▼                │
              │  ┌──────────────────────┐    │
              │  │   RoutingPolicy      │    │
              │  │  WeightedRuleBased   │    │
              │  │  или SLMRouting      │    │
              │  └──────────┬───────────┘    │
              └─────────────┼────────────────┘
                            │ "_slm" / "_llm"
               ┌────────────┴────────────┐
               ▼                         ▼
         ┌──────────┐             ┌──────────┐
         │   SLM    │             │   LLM    │
         │ (<10B)   │             │  (>35B)  │
         └────┬─────┘             └────┬─────┘
              └────────────┬───────────┘
                           │ InferenceRecord
                           ▼
              ┌──────────────────────────────┐
              │      RAGPipelineRunner       │
              │       .aevaluate()           │
              │                              │
              │  BERTScore + ROUGE           │
              │  + LLM Judge (JScore)        │
              └──────────────────────────────┘
                           │ EvaluationRecord
                           ▼
              ┌──────────────────────────────┐
              │  compute_slm_routing_metrics │
              │  get_evaluation_summary      │
              └──────────────────────────────┘
```

### Модульная структура

| Модуль | Ответственность |
|--------|----------------|
| `core/messaging/` | Реестр промптов, рендеринг шаблонов, структурированный разбор ответов |
| `core/data/` | Асинхронная генерация синтетических данных, загрузка датасета |
| `core/tasks/` | Единица инференса: запрос + документы → предсказание |
| `core/router/` | Извлечение признаков, политики маршрутизации, диспетчер |
| `core/pipeline/` | Оркестрация эксперимента, многоуровневая оценка качества |
| `core/utils/` | Постобработка метрик, агрегация результатов |

---

## Генерация синтетических данных

### Пространство датасета

Синтетический датасет организован по трём измерениям — каждый пример однозначно идентифицируется комбинацией task × domain × difficulty:

| Измерение | Значения |
|-----------|---------|
| Задача (`task`) | `reranking`, `context_compression`, `question_answering` |
| Домен (`domain`) | `coding`, `history`, `math`, `medicine`, `research` |
| Сложность (`difficulty`) | `easy`, `medium`, `complex` |

Декларация датасета задаётся через `DatasetDeclaration`:

```python
declaration = DatasetDeclaration(
    tasks=["reranking", "context_compression"],
    domains=["coding", "history", "math", "medicine", "research"],
    difficulties=["easy", "medium", "complex"],
    batch_size=10  # примеров на комбинацию
)
# declaration.n_samples == 2 × 5 × 3 × 10 = 300
```

### Асинхронный генератор

`RAGDatasetAsyncGenerator` реализует конкурентную генерацию с автоматическим retry — если LLM вернул невалидный JSON, пример просто пропускается, а пул пополняется до нужного размера:

```
для каждой (task, domain, difficulty):
    while len(сгенерировано) < batch_size:
        вызвать LLM конкурентно для (batch_size - len(сгенерировано)) запросов
        успешно распарсенные → сохранить в JSON
        OutputParserException → залогировать, пропустить
```

Каждый успешно сгенерированный пример немедленно сохраняется в файл:

```
slm_flow_df/
├── reranking/
│   ├── coding/
│   │   ├── easy/
│   │   │   ├── 3f2a1b4c-....json
│   │   │   └── ...
│   │   ├── medium/
│   │   └── complex/
│   └── ...
├── context_compression/
│   └── ...
└── question_answering/
    └── ...
```

Пример JSON-записи для задачи `context_compression`:

```json
{
  "query": "How do transformer attention mechanisms handle long-range dependencies?",
  "documents": [
    {"idx": 0, "content": "Self-attention computes pairwise...", "reasoning_trace": null},
    {"idx": 1, "content": "Recurrent networks process tokens...", "reasoning_trace": null}
  ],
  "golden_answer": "Transformer attention directly connects any two positions...",
  "optimal_compression_length": 80
}
```

### Pydantic-модели генерации

Все выходы LLM при генерации данных валидируются через Pydantic-схемы:

- `RerankingSample` — запрос, список документов с `reasoning_trace`, `golden_answer` (индекс релевантного документа)
- `CompressionSample` — то же плюс `optimal_compression_length` (целевая длина в словах)

### Загрузка внешних данных: RAGBenchDataset

Помимо синтетических данных, фреймворк поддерживает загрузку реальных бенчмарков. `RAGBenchDataset` читает Parquet-файл (формат RAGBench) и приводит его к стандартному `DatasetRecord` с задачей `question_answering`:

```python
dataset = RAGBenchDataset.from_files("path/to/ragbench.parquet")
```

Все записи получают `task="question_answering"`, `domain="tech"`, `difficulty="abstract"`. Это позволяет прогнать внешний бенчмарк через тот же пайплайн без каких-либо изменений в коде.

---

## Промпт-инжиниринг и структурированный вывод

### Проблема: нестабильные ответы LLM

Языковые модели склонны отвечать в произвольном формате. В пайплайне с несколькими последовательными LLM-вызовами это особенно больно: если один шаг вернул неожиданный формат, вся цепочка ломается. Поэтому каждый вызов в фреймворке требует строго структурированного ответа.

### Решение: LangchainMessageBuilder + PydanticOutputParser

```
Шаблон промпта + {fmt}
         ↓
LangchainMessageBuilder.create_message(key, **kwargs)
         ↓
SystemMessage с injected format instructions
         ↓
LLM API call
         ↓
PydanticOutputParser.parse(response)  ← try/except OutputParserException
         ↓
Типизированный Pydantic-объект
```

`{fmt}` в шаблоне заменяется на инструкции формата, сгенерированные `PydanticOutputParser.get_format_instructions()`. Ожидаемая схема ответа описывается прямо в промпте — это самый надёжный способ добиться стабильного JSON от модели.

### Реестр промптов

`PROMPT_REGISTRY` — замороженный dataclass, предоставляющий именованный доступ ко всем промптам системы:

| Ключ | Назначение |
|------|-----------|
| `reranking_data_generation` | Генерация reranking-примеров |
| `context_compression_data_generation` | Генерация compression-примеров |
| `reranking_inference` | Инференс reranking |
| `context_compression_inference` | Инференс context compression |
| `question_answering_inference` | Инференс question answering |
| `evaluation` | LLM-судья |
| `slm_as_router` | SLM-роутер: оценка сложности запроса |

### Инициализация builder'а

```python
message_builder = LangchainMessageBuilder.from_sequence(
    ("reranking", PROMPT_REGISTRY.reranking_inference, RAGTaskPrediction),
    ("context_compression", PROMPT_REGISTRY.context_compression_inference, RAGTaskPrediction),
    ("question_answering", PROMPT_REGISTRY.question_answering_inference, RAGTaskPrediction),
    ("judge", PROMPT_REGISTRY.evaluation, JScore),
)
```

---

## Извлечение признаков

Чтобы маршрутизатор мог принять решение без участия LLM, нужно численно описать сложность задачи. `RAGFeatureExtractor` делает именно это — превращает запрос и документы в компактный числовой вектор.

### Инструменты извлечения

- **spaCy** (`en_core_web_lg`) — NLP: noun chunks, лемматизация, векторные представления для косинусного сходства
- **tiktoken** (`o200k_base`) — подсчёт токенов в запросе и документах
- **wordfreq** — средняя частотность слов в запросе (мера специализированности лексики)

### Общие признаки (все векторы)

| Признак | Описание | Интерпретация |
|---------|---------|---------------|
| `query_token_count` | Количество токенов в запросе | Длиннее запрос → сложнее задача |
| `query_noun_chunk_count` | Количество именных групп | Много сущностей → сложная предметная область |
| `query_avg_word_frequency` | Средняя частотность слов | Низкая частота → специализированная лексика |
| `avg_lexical_overlap` | Средний лексический overlap запрос/документы | Низкий overlap → сложнее найти релевантное |

### Признаки для Reranking (`RerankingVector`)

| Признак | Описание | Интерпретация |
|---------|---------|---------------|
| `min_lexical_overlap` | Минимальный overlap по всем документам | Очень низкий минимум → есть явно нерелевантные документы |
| `documents_cosine_similarity` | Среднее попарное косинусное сходство документов | Высокое сходство → документы похожи, реранкинг труднее |
| `documents_count` | Количество документов | Больше документов → больше пространство поиска |

### Признаки для Context Compression (`CompressionVector`)

| Признак | Описание | Интерпретация |
|---------|---------|---------------|
| `total_context_token_count` | Суммарный размер контекста в токенах | Большой контекст → сложнее компрессия |
| `avg_chunk_token_count` | Средний размер документа в токенах | Длинные документы → больше информации для фильтрации |
| `relevant_documents_ratio` | Доля документов с overlap ≥ 0.3 | Мало релевантных → нужно внимательнее выбирать |

### Признаки для Question Answering (`QuestionAnsweringVector`)

| Признак | Описание | Интерпретация |
|---------|---------|---------------|
| `max_lexical_overlap` | Максимальный overlap по всем документам | Если ни один документ не перекрывается — задача сложная |
| `relevant_documents_ratio` | Доля документов с overlap ≥ 0.3 | Мало релевантных → ответ разбросан по документам |
| `max_semantic_similarity` | Максимальное косинусное сходство запроса с любым документом | Низкое максимальное сходство → ни один документ явно не отвечает |
| `documents_count` | Количество документов | Больше документов → сложнее выбрать нужное |

### Алгоритм лексического overlap

Для каждого документа:
1. Токенизировать запрос и документ через spaCy
2. Оставить только алфавитные, не стоп-слова, не пунктуацию
3. Лемматизировать оба множества
4. `overlap = |лемм запроса ∩ лемм документа| / |лемм запроса|`

---

## Политики маршрутизации

Все политики реализуют единый структурный протокол `Routable`:

```python
class Routable(Protocol):
    async def call_large_model(self, features: TRoutableFeatures) -> bool: ...
```

`True` означает «направить к LLM», `False` — «достаточно SLM».

### Стратегия 1: WeightedRuleBasedRoutingPolicy

Детерминированные правила над признаковым вектором. Каждое правило — пороговое условие с весом:

```python
WeightedRule(
    name="query_avg_word_frequency",  # имя признака
    operator="le",                     # оператор: gt/ge/lt/le/eq
    threshold=3.0,                     # пороговое значение
    weight=0.20,                       # вес при срабатывании
)
```

Решение принимается по двойному критерию — оба условия должны выполниться одновременно:

1. Сработало не менее `min_triggers` правил
2. Сумма весов сработавших правил ≥ `cumulative_weights_threshold`

Двойной критерий защищает от ложных срабатываний: один сильный сигнал не может переопределить решение без подтверждения других признаков.

**Пример — правила для задачи reranking:**

```python
reranking_task_rules = [
    WeightedRule("query_token_count",           "ge", 55,   weight=0.10),
    WeightedRule("query_noun_chunk_count",       "ge", 7,    weight=0.15),
    WeightedRule("query_avg_word_frequency",     "le", 3.0,  weight=0.20),
    WeightedRule("avg_lexical_overlap",          "le", 0.10, weight=0.25),
    WeightedRule("min_lexical_overlap",          "le", 0.05, weight=0.15),
    WeightedRule("documents_cosine_similarity",  "ge", 0.85, weight=0.25),
    WeightedRule("documents_count",              "ge", 10,   weight=0.10),
]

policy = WeightedRuleBasedRoutingPolicy(
    *reranking_task_rules,
    min_triggers=3,
    cumulative_weights_threshold=0.65
)
```

Правила для `context_compression` дополнительно учитывают `total_context_token_count`, `avg_chunk_token_count` и `relevant_documents_ratio`. Для `question_answering` — `max_lexical_overlap`, `relevant_documents_ratio`, `max_semantic_similarity`.

**Преимущества:** полная интерпретируемость, нулевая латентность (нет дополнительного LLM-вызова), легко настраивать пороги.

**Ограничения:** признаки взаимосвязаны нелинейно; правила не адаптируются к изменению распределения входных данных.

### Стратегия 2: SLMRoutingPolicy

Вместо ручных правил — малая модель как эксперт по оценке сложности. SLM получает на вход запрос и документы, возвращает оценку по шкале Ликерта 1–5:

```python
class SLMRouterOutput(BaseModel):
    confidence: int  # 1 = явно SLM справится, 5 = необходим LLM
```

Решение: если `confidence >= threshold` (по умолчанию 4), направить к LLM.

```python
slm_as_router_policy = SLMRoutingPolicy(
    client=ChatOpenAI(model="mistralai/ministral-3b-2512", ...),
    message_builder=slm_routing_policy_messages_builder,
    confidence_threshold=4
)
```

При `OutputParserException` политика консервативно возвращает `confidence=0` — лучше пропустить сложный запрос через SLM, чем сломать пайплайн.

**Преимущества:** не требует ручного конструирования признаков; адаптируется к семантической сложности.

**Ограничения:** добавляет один LLM-вызов к каждому запросу; качество зависит от способности роутера-SLM корректно калибровать сложность.

### LMRouter: три режима работы

```python
class LMRouter:
    async def select_language_model(
        self, task_instance: RAGTask
    ) -> tuple[TFeatureVector, str]: ...
```

| Режим | Поведение | Возвращает |
|-------|-----------|-----------|
| `"slm"` | Всегда малая модель | `"_slm"` |
| `"llm"` | Всегда большая модель | `"_llm"` |
| `"dynamic"` | Политика для `task.name` определяет выбор | `"_slm"` или `"_llm"` |

В режиме `"dynamic"` роутер автоматически определяет тип политики: `WeightedRuleBasedRoutingPolicy` принимает **признаковый вектор**, `SLMRoutingPolicy` — **сырой `RAGTask`** (для рендеринга промпта).

---

## Пайплайн инференса и оценки

### RAGPipelineRunner: оркестратор эксперимента

```python
runner = RAGPipelineRunner(
    small_model="mistralai/ministral-8b-2512",
    large_model="qwen/qwen3.5-122b-a10b",
    judge_model="openai/gpt-5-mini",
    routing_mode="dynamic",
    messages_builder=message_builder,
    dynamic_routing_policies={
        "reranking": WeightedRuleBasedRoutingPolicy(...),
        "context_compression": SLMRoutingPolicy(...),
        "question_answering": WeightedRuleBasedRoutingPolicy(...),
    },
    extractor_spacy_nlp="en_core_web_lg",
    extractor_tokenizer_name="o200k_base",
    model_kwargs=model_kwargs,
)
```

### arun(): параллельный инференс

```
1. tasks = [RAGTask.from_record(r) for r in dataset]

2. Routing phase (конкурентно):
   routing_results = await asyncio.gather(*[
       router.select_language_model(task) for task in tasks
   ])

3. Inference phase (конкурентно):
   predictions = await asyncio.gather(*[
       task.agenerate_prediction(slm_client if route == "_slm" else llm_client, ...)
       for task, (fvector, route) in zip(tasks, routing_results)
   ])

4. Сборка InferenceRecord для каждого примера
```

### aevaluate(): многоуровневая оценка

Для каждого `InferenceRecord` параллельно вычисляются автоматические метрики и запускается LLM-судья.

**Автоматические метрики по задачам:**

| Задача | Метрики |
|--------|---------|
| `reranking` | BERTScore F1, exact match (строгое совпадение с golden_answer) |
| `context_compression` | BERTScore F1, ROUGE-L, ROUGE-2, compression ratio |
| `question_answering` | BERTScore F1, ROUGE-L |

`compression_ratio = tokens(prediction) / total_context_tokens` — насколько эффективно модель сжала контекст.

**LLM-судья (JScore):**

```python
class JScore(BaseModel):
    feedback: str           # развёрнутый анализ ответа
    factual_precision: int  # 1–5: точность фактов
    completeness: int       # 1–5: полнота ответа
    hallucination: int      # 1–5: отсутствие галлюцинаций (5 = нет галлюцинаций)
    final_score: int        # = min(factual_precision, completeness, hallucination)
```

`final_score = min(...)` — консервативная агрегация: один слабый критерий тянет итоговую оценку вниз. Это важно: модель не должна получать высокий балл за полноту, если при этом галлюцинирует.

Судья получает контекст о задаче из `TASK_DESCRIPTIONS` — это позволяет оценивать reranking, compression и QA по релевантным для каждого типа критериям.

### Постобработка: метрики маршрутизации и агрегация

```python
metrics = compute_slm_routing_metrics(
    records=evaluation_records,
    threshold=4.0  # jscore.final_score >= threshold = "успешный SLM-вызов"
)
```

```python
class SLMRoutingMetrics(BaseModel):
    slm_routing_ratio: float   # доля примеров, направленных к SLM
    slm_success_ratio: float   # среди SLM-вызовов: доля успешных (jscore >= threshold)
```

Эти два числа — ключевой trade-off эксперимента: `slm_routing_ratio` — экономия ресурсов, `slm_success_ratio` — качество при этой экономии.

Для агрегированной картины по всему прогону используется `get_evaluation_summary()`:

```python
summary = get_evaluation_summary(evaluation_records)
# → EvaluationSummary:
#   reranking_avg_bert_f1, reranking_avg_exact_match, reranking_avg_jscore
#   compression_avg_bert_f1, compression_avg_rouge_l, compression_avg_rouge_n,
#   compression_avg_compression_ratio, compression_avg_jscore
#   question_answering_avg_jscore, question_answering_avg_bert_f1,
#   question_answering_avg_rouge_l
```

---

## Технические решения

### Конкурентность без перегрузки API

Все LLM-вызовы ограничены через `InMemoryRateLimiter`:

```python
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=0.1,
    max_bucket_size=10
)
```

### Обработка ошибок парсинга

Все LLM-вызовы оборачиваются в `try/except OutputParserException`. Поведение при ошибке зависит от контекста:

- **Генерация данных:** пример пропускается, loop повторяет попытку
- **Инференс:** возвращается `RAGTaskPrediction(content="structured_output_parsing_error")`
- **SLM-роутер:** консервативно возвращается `confidence=0` (направить к SLM)

Такое поведение предотвращает каскадные ошибки при сохранении работоспособности всего пайплайна.

### Загрузка датасета

`RAGSyntheticDataset.from_files()` рекурсивно обходит директорию `slm_flow_df/`, восстанавливая метаданные из структуры пути (task, domain, difficulty). `RAGBenchDataset.from_files()` читает Parquet-файл и делает то же самое для внешних данных. Оба класса поддерживают слайсинг и автоматическую перемешку после загрузки.

---

## Расширяемость

Фреймворк спроектирован так, чтобы добавление нового компонента не ломало существующий код:

| Что добавить | Что реализовать |
|-------------|----------------|
| Новая политика маршрутизации | Реализовать протокол `Routable` с методом `call_large_model()` |
| Новая задача RAG | Добавить промпт в `PROMPT_REGISTRY`, вектор признаков в `RAGFeatureExtractor`, метрики в `RAGPipelineRunner` |
| Новые метрики оценки | Расширить соответствующий `_compute_*_metrics()` метод в `RAGPipelineRunner` |
| Новый тип признаков | Унаследоваться от `RAGFeatureVectorBase`, добавить dispatch в `extract_from_task()` |

---

## Зависимости и окружение

**Python:** 3.11+

| Библиотека | Назначение |
|-----------|-----------|
| `langchain`, `langchain-openai` | LLM-интеграция, промпт-рендеринг |
| `pydantic` | Типизированные модели, structured output |
| `spacy` + `en_core_web_lg` | NLP-признаки: noun chunks, лемматизация, cosine similarity |
| `tiktoken` | Подсчёт токенов |
| `wordfreq` | Частотность слов |
| `bert_score` | BERTScore-метрика |
| `rouge` | ROUGE-метрика |
| `aiofiles`, `asyncio` | Асинхронный I/O |
| `pandas` | Табличное представление результатов |

**LLM-провайдер:** [OpenRouter](https://openrouter.ai/) — единая точка доступа к моделям разных вендоров.

```env
OPENROUTER_API_KEY=<your_key>
BASE_URL=https://openrouter.ai/api/v1
```

**Модели, протестированные в экспериментах:**

- SLM: Ministral-8B, Gemma-3-4B, LLaMA-3.2-3B, LLaMA-3.2-1B
- LLM: Qwen3.5-122B, Qwen3.5-35B
- Роутер: Ministral-3B, LLaMA-3.2-1B
- Судья: GPT-5

---

## Выводы

SlmFlowFramework реализует полный цикл эксперимента по динамической маршрутизации SLM/LLM:

1. **Генерация синтетических данных** — покрывает пространство task × domain × difficulty с автоматическим retry и типизированными выходами; внешние датасеты подключаются через `RAGBenchDataset`
2. **Три задачи RAG** — reranking, context compression и question answering, каждая со своим вектором признаков, промптом и метриками
3. **Две комплементарные стратегии маршрутизации** — rule-based (интерпретируемость, нулевая латентность) и SLM-as-router (адаптивность, семантическое понимание сложности)
4. **Многоуровневая оценка** — автоматические метрики (BERTScore, ROUGE) + LLM-судья с консервативной агрегацией + агрегированная сводка через `get_evaluation_summary()`
5. **Модульная архитектура** — каждый компонент заменяем независимо; добавление новой политики или задачи требует минимальных изменений кода
