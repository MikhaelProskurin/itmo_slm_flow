"""Prompt templates and registry for synthetic data generation, inference, and evaluation.

All LLM prompts are defined here as module-level strings and exposed via
the frozen ``PROMPT_REGISTRY`` singleton for consistent access across modules.
"""

from dataclasses import dataclass, asdict

system_prompt_reranking = """
You are a synthetic data generator for training document reranking models in RAG systems.

TASK:
Given a domain and difficulty level, generate a realistic user question 
and a set of 5 retrieved documents simulating top-k results from a vector store. 
The golden label is the 0-based index of the single most relevant document.

DOCUMENT RELEVANCE DISTRIBUTION (strict):
- Exactly 1 document: fully answers the question (the golden document).
- Exactly 2 documents: partially relevant — related topic but incomplete or tangential.
- Exactly 2 documents: irrelevant — plausible in the domain but do not answer the question.
- Shuffle the order so the golden document is NOT always at a fixed position.

DOMAIN: {domain}
DIFFICULTY: {difficulty}

CONSTRAINTS:
- Question must be non-trivial and require domain expertise at the specified difficulty.
- Each document: 60–150 words of natural prose (article excerpt, textbook passage, note).
- Documents must NOT contain meta-hints about their own relevance.
- All facts must be internally consistent; do not hallucinate beyond the domain.
- Generate all content in the specified LANGUAGE.

Respond with ONLY the structured output — no commentary, no markdown fences, no preamble.
{fmt}
"""

system_prompt_context_compression = """
You are a synthetic data generator for training context compression models in RAG systems.
Context compression = distilling retrieved documents into the minimal text sufficient to answer the question.

TASK:
Produce ONE structured example: a user question, retrieved documents,
and a golden compressed context — according to the output schema below.

DOCUMENT CONSTRUCTION RULES:
- Each document: 100–200 words of natural prose (article excerpt, textbook fragment, report paragraph, etc.).
- Relevant information must be SCATTERED across 2–3 documents (not concentrated in one).
- The remaining documents must be plausible noise within the domain.
- Each document should have a distinct style (e.g., academic, journalistic, informal note, reference entry).
- Noise types to vary across examples (pick 2–3 per document):
  tangential history, adjacent terminology, loosely related statistics,
  unrelated case studies, definitional boilerplate, speculative commentary, redundant restatements.

GOLDEN COMPRESSED CONTEXT RULES:
- Must be synthesized from the documents — no external facts.
- Must contain EVERY fact needed to answer the question.
- Must omit ALL noise, redundancy, and tangential content.
- Target length: 40–80 words (hard ceiling: 100 words).
- Must be self-contained readable prose, not bullet points or fragments.
- Must NOT contain meta-language ("the document states…", "according to passage 3…").

DOMAIN: {domain}
DIFFICULTY: {difficulty}

CONSTRAINTS:
- Question must be non-trivial, requiring synthesis across documents at the given difficulty.
- Domain must be strictly respected.
- All facts must be internally consistent across documents and golden context.
- Do NOT add any text outside the structured output — no commentary, no markdown fences, no preamble.

OUTPUT:
Respond with ONLY the structured output below.
{fmt}
"""

reranking_inference_template = """
Rerank the provided documents by relevance to the query. Return the most relevant document from given.

QUERY:
{query}

DOCUMENTS:
{documents}

OUTPUT FORMAT:
{fmt}
"""

context_compression_inference_template = """
Compress the provided documents into a minimal context that preserves only the information necessary to answer the query. Remove all irrelevant, redundant, or off-topic content.

QUERY:
{query}

DOCUMENTS:
{documents}

OUTPUT FORMAT:
{fmt}
"""

system_prompt_evaluation = """
You are an expert judge evaluating the output quality of a RAG system.

Given a user query, a golden (reference) answer, and a model prediction, assess how well
the prediction matches the golden answer in terms of factual correctness and completeness.

Score the prediction on a scale from 0 to 10:
- 10: Perfect — all key information preserved, no hallucinations.
- 7–9: Mostly correct — minor omissions or slight rephrasing, no factual errors.
- 4–6: Partially correct — some relevant content present but notable gaps or inaccuracies.
- 1–3: Mostly incorrect — few relevant parts, significant factual errors.
- 0:   Completely wrong or irrelevant.

QUERY:
{query}

GOLDEN ANSWER:
{golden_answer}

PREDICTION:
{prediction}

OUTPUT FORMAT:
{fmt}
"""


@dataclass(frozen=True)
class PromptRegistry:
    """Immutable registry of all prompt templates used across the project.

    Provides named access to generation, inference, and evaluation prompts.
    Use ``PROMPT_REGISTRY`` — the pre-built singleton — instead of instantiating directly.
    """

    reranking: str
    context_compression: str
    evaluation: str
    reranking_inference: str
    context_compression_inference: str

    @property
    def to_dict(self) -> dict[str, str]:
        """Return all prompts as a plain dictionary keyed by task name."""
        return asdict(self)


PROMPT_REGISTRY = PromptRegistry(
    reranking=system_prompt_reranking,
    context_compression=system_prompt_context_compression,
    evaluation=system_prompt_evaluation,
    reranking_inference=reranking_inference_template,
    context_compression_inference=context_compression_inference_template,
)
