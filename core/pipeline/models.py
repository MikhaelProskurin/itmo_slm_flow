from pydantic import BaseModel, Field

class JudgeVerdict(BaseModel):
    """Structured output from the LLM-as-a-judge evaluator."""

    feedback: str = Field(
        description=(
            "Detailed critical analysis covering factual precision, "
            "completeness, and hallucination criteria. "
            "Must be written BEFORE assigning any scores."
        ),
    )
    factual_precision: int = Field(description="How accurately the response conveys facts compared to the reference (1-5).")
    completeness: int = Field(description="How well the response covers all key information from the reference (1-5).")
    hallucination: int = Field(description="Degree to which the response is free of unsupported claims (1-5).")
    final_score: int = Field(description="Minimum of factual_precision, completeness, and hallucination scores.")


class FlowResult(BaseModel):
    """Single inference + evaluation record produced by InferenceFlow."""

    task: str
    routed_model: str
    query: str
    golden_answer: str
    prediction: str
    features: RerankingFeatures | ContextCompressionFeatures | None
    judge_score: float | None
    judge_reasoning: str | None