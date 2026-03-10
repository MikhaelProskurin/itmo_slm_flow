"""Utility functions for the inference flow: judge evaluation, metric aggregation, and timing.

Provides an async LLM-as-a-judge helper, BERTScore/ROUGE-2 aggregation over
``FlowResult`` lists, and a ``@timeit`` decorator for wall-clock profiling.
"""

import time
from functools import wraps
from typing import Callable

import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from bert_score import score
from rouge import Rouge

from core.io.models import JudgeVerdict, FlowResult


async def run_judge_llm(client: ChatOpenAI, template: str, **kwargs) -> JudgeVerdict | None:
    """Invoke the LLM judge asynchronously and return a parsed verdict, or ``None`` on parse failure.

    param: client: Async-capable LangChain LLM client used for the judge call.
       type: ChatOpenAI
    param: template: Evaluation prompt template; must contain ``{fmt}`` plus any keys in ``kwargs``.
       type: str
    param: **kwargs: Template variables (``query``, ``golden_answer``, ``prediction``).
       type: Any
    """
    parser = PydanticOutputParser(pydantic_object=JudgeVerdict)
    kwargs |= {"fmt": parser.get_format_instructions()}
    prompt = [SystemMessage(template.format(**kwargs))]
    response: AIMessage = await client.ainvoke(prompt)
    try:
        return parser.parse(response.content)
    except OutputParserException:
        return None


def compute_routing_metrics(scored_by_judge: list[FlowResult], slm_name: str) -> dict[str, float]:
    """Compute SLM routing effectiveness metrics over a set of judge-evaluated flow results.

    Returns a dict with two keys:

    - ``slm_success_rate``: fraction of SLM-routed calls where ``judge_score > 8.0``.
    - ``slm_routing_ratio``: fraction of all calls that were routed to the SLM.

    param: scored_by_judge: List of evaluated flow results; each must have a valid ``judge_score``.
       type: list[FlowResult]
    param: slm_name: Model identifier used to distinguish SLM-routed rows (matched against ``routed_model``).
       type: str
    """
    slm_calls = [row for row in scored_by_judge if row.routed_model == slm_name]
    success_slm_calls = [call for call in slm_calls if call.judge_score > 8.0]

    return {
        "slm_success_rate": len(success_slm_calls) / len(slm_calls),
        "slm_routing_ratio": len(slm_calls) / len(scored_by_judge)
    }


def get_reranking_metrics(batch: list[FlowResult]) -> dict[str, float]:
    """Compute BERTScore-F1 and mean judge score for a batch of reranking results.

    param: batch: Judge-evaluated flow results for the reranking task.
       type: list[FlowResult]
    """
    bert_f1_score = score(
        cands=[row.prediction for row in batch],
        refs=[row.golden_answer for row in batch],
        lang="en",
        rescale_with_baseline=True
    )
    return {
        "bert_f1_score": bert_f1_score[-1].mean().item(),
        "judge_scores": sum(row.judge_score for row in batch) / len(batch)
    }


def get_context_compression_metrics(batch: list[FlowResult]) -> dict[str, float]:
    """Compute BERTScore-F1, ROUGE-2, compression ratio, and mean judge score for a batch of compression results.

    ``compression_ratio`` is the mean ratio of prediction token count to original context token count
    (from ``ContextCompressionFeatures.total_context_token_count``); lower values indicate more aggressive compression.

    param: batch: Judge-evaluated flow results for the context compression task.
       type: list[FlowResult]
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    rouge = Rouge()
    
    bert_f1_score = score(
        cands=[row.prediction for row in batch],
        refs=[row.golden_answer for row in batch],
        lang="en",
        rescale_with_baseline=True
    )
    rouge_scores = [
        rouge.get_scores(hyps=row.prediction, refs=row.golden_answer, avg=True)["rouge-2"]["f"]
        for row in batch
    ]
    compression_ratios = [
        len(tokenizer.encode(row.prediction)) / row.features.total_context_token_count
        for row in batch
    ]
    return {
        "bert_f1_score": bert_f1_score[-1].mean().item(),
        "rouge_2_score": sum(rouge_scores) / len(rouge_scores),
        "compression_ratio": sum(compression_ratios) / len(compression_ratios),
        "judge_scores": sum(row.judge_score for row in batch) / len(batch)
    }


def compute_stat_metrics(scored_by_judge: list[FlowResult], task_scoring_map: dict[str, Callable]) -> list[dict[str, dict]]:
    """Compute task-specific metrics for all tasks present in the results.

    Groups results by task name and dispatches each group to the corresponding scoring
    function in ``task_scoring_map``. Returns one metrics dict per distinct task.

    param: scored_by_judge: List of evaluated flow results produced by ``InferenceFlow.evaluate_by_judge``.
       type: list[FlowResult]
    param: task_scoring_map: Mapping from task name to its metric-computing function (e.g. ``{"reranking": get_reranking_metrics}``).
       type: dict[str, Callable]
    """
    distinct_tasks = {result.task for result in scored_by_judge}
    batches_by_task = [(name, [r for r in scored_by_judge if r.task == name]) for k in distinct_tasks]

    results = []
    for name, batch in batches_by_task:
        results.append({
            "task": name, 
            "results": task_scoring_map[name](batch)
        })

    return results


def timeit(func: Callable):
    """Decorator that prints the wall-clock runtime of the wrapped function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"runtime: {end - start}")
        return result
    return wrapper
