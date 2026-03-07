"""Prompt templates and registry for synthetic data generation, inference, and evaluation.

All LLM prompts are defined here as module-level strings and exposed via
the frozen ``PROMPT_REGISTRY`` singleton for consistent access across modules.
"""

from dataclasses import dataclass, asdict

system_prompt_reranking = """
You are a synthetic data generator for Retrieval-Augmented Generation (RAG) systems.
Your role is to generate high-quality datasets for training and evaluating document reranking models.

TASK:
Generate a single example consisting of:
1. A user question in the specified domain and difficulty.
2. Exactly 5 retrieved documents (simulating vector database output) with varying relevance:
   - 1 document that is clearly the most relevant and fully answers the question
   - 2 documents that are partially relevant or incomplete
   - 2 documents that are irrelevant but plausible
3. A golden answer is the most relevant document from the list of documents received.

DOMAIN: {domain}
DIFFICULTY: {difficulty}

REQUIREMENTS:
- The user question must be realistic and non-trivial.
- The retrieved documents must look like natural text passages (articles, notes, explanations, etc.).
- Documents should vary in relevance:
  - some directly answer the question
  - some are tangentially related
  - some are misleading or off-topic but plausible
- Do NOT mention which documents are relevant explicitly in the text of the document.
- Do NOT include any explanations or commentary outside the structured output.
- Do NOT explicitly label or mention which document is relevant in the document texts.
- Do NOT include phrases such as "this document answers the question" or similar indicators of relevance.
- The golden answer is the most relevant document.

QUALITY CONSTRAINTS:
- Domain must be strictly respected (e.g., if domain = math, use mathematical content).
- Difficulty must match the requested level (easy / medium / hard).
- Avoid trivial QA pairs.
- Ensure internal consistency between documents and golden answer.
- Do not hallucinate facts outside the provided documents.

Generate exactly one example per response.

OUTPUT FORMAT:
Use the following structured output format exactly:
{fmt}
"""

system_prompt_context_compression = """
You are a synthetic data generator for Retrieval-Augmented Generation (RAG) systems.
Your role is to generate datasets for training and evaluating context compression (context distillation / summarization for QA).

TASK:
Generate a single example consisting of:
1. A user question in the specified domain and difficulty.
2. Exactly 5 retrieved documents (simulating vector database output) forming a long context, where:
    - only part of the content is relevant to answering the question
    - the rest is noisy, redundant, or loosely related
3. A golden answer it's a compressed context that preserves only the minimal information necessary to answer the question.

DOMAIN: {domain}
DIFFICULTY: {difficulty}

REQUIREMENTS:
- The user question must be realistic and non-trivial.
- The 5 retrieved documents must be verbose and heterogeneous in style and content.
- Include irrelevant paragraphs such as:
    - historical background
    - side examples
    - definitions not needed for answering
    - loosely related concepts
- The golden answer must:
    - remove all irrelevant information
    - preserve all facts strictly necessary to answer the question
    - be significantly shorter than the combined original documents
- Do NOT introduce any external knowledge beyond the provided documents.
- Do NOT add explanations or commentary outside the structured output.

QUALITY CONSTRAINTS:
- Domain must be strictly respected.
- Difficulty level must match the requested level (easy / medium / hard).
- Compression must be meaningful (not a trivial copy-paste).
- Ensure consistency between:
    - the user question
    - the retrieved documents
    - the golden answer
- Do not hallucinate facts not present in the retrieved documents.
- Avoid meta statements (e.g., "this document shows", "the answer is").

Generate exactly one example per response.

OUTPUT FORMAT:
Use the following structured output format exactly:
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
