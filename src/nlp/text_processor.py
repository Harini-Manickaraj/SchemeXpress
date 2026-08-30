"""
SchemeXpress NLP text preprocessing utilities.

This module provides conservative text preprocessing for government
scheme descriptions before TF-IDF vectorization.
"""

import re
import unicodedata


def normalize_text(text):
    """
    Normalize scheme text while preserving useful natural-language content.

    Steps:
    - Handle missing/non-string values
    - Unicode normalization
    - Convert to lowercase
    - Replace URLs with a space
    - Remove unwanted punctuation
    - Normalize whitespace
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # URLs are generally not useful for semantic similarity.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters, numbers and whitespace.
    # This preserves useful terms such as age ranges and income values.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_corpus(texts):
    """
    Apply normalize_text to an iterable/Series of documents.
    """
    return texts.fillna("").map(normalize_text)