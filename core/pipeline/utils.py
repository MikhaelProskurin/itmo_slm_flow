import time
from functools import wraps
from typing import Callable


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
