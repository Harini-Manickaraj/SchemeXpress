"""
TF-IDF based scheme similarity engine for SchemeXpress.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processor import normalize_text


class SchemeTfidfEngine:
    """
    Fits TF-IDF on government scheme text and retrieves
    schemes similar to a user's natural-language requirement.
    """

    def __init__(
        self,
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )

        self.data = None
        self.matrix = None

    def fit(self, dataframe, text_column="combined_text"):
        """
        Fit TF-IDF on the supplied scheme dataframe.
        """
        if text_column not in dataframe.columns:
            raise ValueError(
                f"Required text column '{text_column}' was not found."
            )

        self.data = dataframe.copy()

        texts = self.data[text_column].fillna("").map(normalize_text)

        if texts.str.strip().eq("").all():
            raise ValueError("No usable scheme text was found.")

        self.matrix = self.vectorizer.fit_transform(texts)

        return self

    def search(self, query, top_k=10):
        """
        Return the top-k schemes most similar to a user query.
        """
        if self.matrix is None:
            raise RuntimeError("The TF-IDF engine must be fitted first.")

        if query is None or not str(query).strip():
            raise ValueError("Query cannot be empty.")

        query_text = normalize_text(query)

        query_vector = self.vectorizer.transform([query_text])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = self.data.iloc[ranked_indices].copy()

        results["similarity_score"] = scores[ranked_indices]

        return results.reset_index(drop=True)


def load_cleaned_schemes(csv_path=None):
    """
    Load the Phase 2 cleaned dataset.
    """
    if csv_path is None:
        csv_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "cleaned_schemes.csv"
        )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {csv_path}"
        )

    return pd.read_csv(csv_path)