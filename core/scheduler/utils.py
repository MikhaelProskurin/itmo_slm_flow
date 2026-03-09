"""Low-level NLP helper functions shared by all feature extractors.

Computes query-level statistics (token count, entity count, word frequency)
and query-to-document lexical overlap used as inputs for routing decisions.
"""

import spacy
import tiktoken
from wordfreq import word_frequency


def query_features(query: str, nlp: spacy.Language, tokenizer: tiktoken.Encoding) -> dict[str, float]:
   """Compute shared query-level features used across all feature extractors.

   param: query: Raw query string to analyze.
      type: str
   param: nlp: Loaded spaCy language model for tokenization and noun-chunk extraction.
      type: spacy.Language
   param: tokenizer: tiktoken encoding used for BPE token counting.
      type: tiktoken.Encoding
   """
   doc = nlp(query)
   words = [token.text.lower() for token in doc if token.is_alpha]
   chunks = list(doc.noun_chunks)

   avg_word_freq = (
      sum(word_frequency(w, "en") for w in words) / len(words)
      if words else 0.0
   )

   return {
      "query_token_count": len(tokenizer.encode(query)),
      "query_noun_chunk_count": float(len(chunks)),
      "query_avg_word_frequency": avg_word_freq,
   }


def lexical_overlap(nlp: spacy.Language, query: str, document: str) -> float:
   """Compute the fraction of non-stop query lemmas that also appear in ``document``.

   param: nlp: Loaded spaCy language model for lemmatization and stop-word filtering.
      type: spacy.Language
   param: query: User query string.
      type: str
   param: document: Document text to compare against the query.
      type: str
   """

   nlp_query, nlp_doc = nlp(query), nlp(document)

   query_terms = {
      t.lemma_.lower() for t in nlp_query
      if t.is_alpha and not t.is_stop
   }
   doc_terms = {
      t.lemma_.lower() for t in nlp_doc
      if t.is_alpha and not t.is_stop
   }

   return len(query_terms & doc_terms) / max(len(query_terms), 1)
