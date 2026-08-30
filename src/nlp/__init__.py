# src/nlp/__init__.py
# Natural Language Processing package.
#
# Modules:
#   text_processor.py — Text cleaning, tokenization, stop-word removal, lemmatization
#   tfidf_engine.py   — TF-IDF vectorization and cosine similarity computation



from .text_processor import normalize_text, preprocess_corpus
from .tfidf_engine import SchemeTfidfEngine, load_cleaned_schemes

__all__ = [
    "normalize_text",
    "preprocess_corpus",
    "SchemeTfidfEngine",
    "load_cleaned_schemes",
]