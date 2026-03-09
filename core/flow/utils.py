"""Utility functions for the inference flow: judge evaluation, metric aggregation, and timing.

Provides an async LLM-as-a-judge helper, BERTScore/ROUGE-2 aggregation over
``FlowResult`` lists, and a ``@timeit`` decorator for wall-clock profiling.
"""

import time
from functools import wraps
from typing import Callable

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

def compute_stat_metrics(scored_by_judge: list[FlowResult]) -> tuple[dict[str, float]]:
    """Compute BERTScore-F1 and ROUGE-2 metrics aggregated separately by task type.

    Returns a two-element tuple ``(reranking_dict, compression_dict)`` where each dict
    contains ``bert_score_f1``, ``judge_scores``, and (for compression) ``rouge_2_score``.

    param: scored_by_judge: List of evaluated flow results produced by ``InferenceFlow.evaluate_by_judge``.
       type: list[FlowResult]
    """
    rerankings = [result for result in scored_by_judge if result.task == "reranking"]
    compressions = [result for result in scored_by_judge if result.task == "context_compression"]

    reranking_bert_scores = []
    for row in rerankings:

        p, r, f1 = score(
            [row.prediction],
            [row.golden_answer],
            lang="en",
            rescale_with_baseline=True
        )
        reranking_bert_scores.append(f1.mean().item())

    compression_bert_scores = []
    compression_rouge_scores = []
    for row in compressions:

        p, r, f1 = score(
            [row.prediction],
            [row.golden_answer],
            lang="en",
            rescale_with_baseline=True
        )
        rouge_scores = Rouge().get_scores(
            row.prediction,
            row.golden_answer,
            avg=True
        )
        compression_bert_scores.append(f1.mean().item())
        compression_rouge_scores.append(rouge_scores["rouge-2"]["f"])

    reranking_dict = {
        "bert_score_f1": sum(reranking_bert_scores) / len(reranking_bert_scores),
        "judge_scores": sum([r.judge_score for r in rerankings]) / len(rerankings)
    }
    compression_dict = {
        "bert_score_f1": sum(compression_bert_scores) / len(compression_bert_scores),
        "rouge_2_score": sum(compression_rouge_scores) / len(compression_rouge_scores),
        "judge_scores": sum([r.judge_score for r in compressions]) / len(compressions)
    }
    return reranking_dict, compression_dict


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
