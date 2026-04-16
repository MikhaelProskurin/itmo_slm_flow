"""Prompt templates and registry for synthetic data generation, inference, and evaluation.

All LLM prompts are defined here as module-level strings and exposed via
the frozen ``PROMPT_REGISTRY`` singleton for consistent access across modules.
"""

from dataclasses import dataclass, asdict

RERANKING_DATA_GENERATION = """
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


CONTEXT_COMPRESSION_DATA_GENERATION = """
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


RERANKING_INFERENCE = """
Rerank the provided documents by relevance to the query. Return the most relevant document from given.

QUERY:
{query}

DOCUMENTS:
{documents}

OUTPUT FORMAT:
{fmt}
"""


CONTEXT_COMPRESSION_INFERENCE = """
Compress the provided documents into a minimal context that preserves only the information necessary to answer the query. Remove all irrelevant, redundant, or off-topic content.

QUERY:
{query}

DOCUMENTS:
{documents}

OUTPUT FORMAT:
{fmt}
"""


TASK_DESCRIPTIONS = {
    "reranking": (
        "The system performs RERANKING: given a user query and a set of candidate "
        "documents, it reorders them by relevance and returns the top document. "
        "The reference answer contains a SINGLE most relevant document. "
        "Evaluate whether the prediction selected the same document or one with "
        "equivalent informational content. "
        "Factual Precision: does the selected document match the reference document "
        "in content and relevance to the query? "
        "Completeness: does the selected document cover the same key information "
        "as the reference document? A document that is partially relevant but misses "
        "the core answer should score low. "
        "Hallucination: did the system return a document that is irrelevant to the "
        "query or contains information not aligned with the reference?"
    ),
    "context_compression": (
        "The system performs CONTEXT COMPRESSION: given a user query and retrieved "
        "passages, it produces a condensed version retaining only query-relevant "
        "information. Evaluate whether the compressed output preserves all key facts "
        "from the reference. Completeness means no critical information is lost during "
        "compression. Hallucination means the compressed text introduces claims not "
        "present in the original passages."
    ),
}


JUDGE_EVALUATION = """
### Task Description:
You are a strict evaluator assessing a RAG system's output.
You will receive an instruction (user query), a response to evaluate,
and a reference answer that represents a Score 5 response.

**Evaluated Task Type:** {task_type}

**Task Context:**
{task_description}

Apply the evaluation criteria below with this task type and context in mind.
What constitutes a factual error, an omission, or a hallucination
depends on the specific task described above.

### Evaluation Protocol:

1. Write a detailed feedback analyzing the response against EACH of the
   criteria below. Be critical — identify every omission, error, and
   hallucination explicitly. Do NOT give the benefit of the doubt.
2. After writing the feedback, assign a score (1-5) for each criterion.
3. Then compute the final score as the MINIMUM of the individual scores.
   (One bad dimension caps the overall quality.)
4. Follow the output format specified below exactly.

### Evaluation Criteria:

**Criterion A: Factual Precision**
How accurately does the response convey facts compared to the reference?
Score 1: Contains multiple critical factual errors or fabrications.
Score 2: Contains at least one significant factual error.
Score 3: Facts are mostly correct but imprecise or partially distorted.
Score 4: All stated facts are correct; minor imprecision in wording.
Score 5: All facts are accurate and precisely stated, matching the reference.

**Criterion B: Completeness**
Does the response cover all key information from the reference answer?
Score 1: Misses nearly all key points from the reference.
Score 2: Covers less than half of the key points.
Score 3: Covers roughly half of the key points; notable gaps remain.
Score 4: Covers most key points; only minor details are missing.
Score 5: Covers all key points from the reference with no omissions.

**Criterion C: Hallucination / Extraneous Content**
Does the response add claims not supported by the reference?
Score 1: Predominantly hallucinated or fabricated content.
Score 2: Contains multiple unsupported claims that mislead the reader.
Score 3: Contains at least one unsupported claim, but core content is grounded.
Score 4: No hallucinations; may include minor, harmless elaborations.
Score 5: Strictly grounded — no information beyond what the reference supports.

### User Query:
{query}

### Response to Evaluate:
{prediction}

### Reference Answer (Score 5):
{golden_answer}

### Output Format:
{fmt}
"""


SLM_AS_ROUTER = """
You are a routing judge for a RAG pipeline.
You receive a task, a query, and retrieved documents. You estimate how strongly THIS instance requires a Large Language Model (LLM) instead of a Small Language Model (SLM).

You judge the task itself — not your own certainty about any answer.

## Scale (1–5 Likert)
- 1 — Trivial for SLM. Clear query, answer localized in one document, pure extraction or obvious ranking.
- 2 — Mostly easy for SLM. Slight ambiguity or minor distractors, but no real reasoning required.
- 3 — Borderline. Moderate ambiguity, some cross-document alignment, or mild domain specificity.
- 4 — Likely needs LLM. Multi-hop reasoning, contradictions to resolve, dense domain notation, or synthesis beyond extraction.
- 5 — Clearly needs LLM. Heavy multi-hop chains, conflicting evidence, abstract or underspecified query, or long scattered context requiring unification.

## Signals for task = reranking
Raise the score when:
- Relevance depends on subtle semantic distinctions rather than lexical overlap.
- Several documents look superficially relevant but only some actually answer the query.
- Query is abstract, comparative, or requires understanding intent beyond keywords.

Lower the score when:
- One document obviously matches the query both lexically and topically.
- Candidate documents cover clearly different topics and are easy to separate.

## Signals for task = context_compression
Raise the score when:
- Key information is scattered across many documents and must be unified.
- Documents contain contradictions that must be resolved before compressing.
- Query demands synthesis, causal explanation, or comparison.
- Relevant facts are interleaved with closely related distractors.

Lower the score when:
- Relevant content is concentrated in a small contiguous span of a single document.
- Documents are short and the query is narrowly extractive.

## Calibration
- Commit to exactly one integer from 1 to 5. Do not output ranges, decimals, or hedged values.
- Do not default to 3 when unsure — pick the nearest level that best matches the signals.
- Use the full scale across instances; avoid clustering only on 1 and 5.

## Input

### task
{name}

### query
{query}

### documents
{documents}

## Output
Produce only the JSON object described below — no preamble, no reasoning text.

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
    slm_as_router: str

    @property
    def to_dict(self) -> dict[str, str]:
        """Return all prompts as a plain ``dict`` keyed by field name."""
        return asdict(self)


PROMPT_REGISTRY = PromptRegistry(
    reranking_data_generation=RERANKING_DATA_GENERATION,
    context_compression_data_generation=CONTEXT_COMPRESSION_DATA_GENERATION,
    evaluation=JUDGE_EVALUATION,
    reranking_inference=RERANKING_INFERENCE,
    context_compression_inference=CONTEXT_COMPRESSION_INFERENCE,
    slm_as_router=SLM_AS_ROUTER
)
