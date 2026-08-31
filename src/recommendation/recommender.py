# src/recommendation/recommender.py

from typing import Any

from src.nlp.tfidf_engine import SchemeTfidfEngine
from .eligibility_engine import EligibilityEngine


class SchemeRecommender:
    """
    Hybrid scheme recommender.

    Uses:
        1. TF-IDF similarity for relevance.
        2. EligibilityEngine for preliminary eligibility filtering.

    Clear eligibility mismatches are excluded from recommendations.
    UNKNOWN eligibility is retained but clearly marked.
    """

    def __init__(self, tfidf_engine=None, eligibility_engine=None):
        self.tfidf_engine = tfidf_engine or SchemeTfidfEngine()
        self.eligibility_engine = eligibility_engine or EligibilityEngine()

    def fit(self, dataframe, text_column="combined_text"):
        """Fit the TF-IDF engine on scheme data."""
        self.tfidf_engine.fit(
            dataframe,
            text_column=text_column,
        )

        return self

    def recommend(
        self,
        query: str,
        user_profile: dict[str, Any],
        top_k: int = 10,
        candidate_k: int = 50,
    ):
        """
        Retrieve relevant schemes and remove clear eligibility mismatches.

        candidate_k:
            Number of TF-IDF candidates examined before eligibility filtering.
        """

        candidates = self.tfidf_engine.search(
            query,
            top_k=candidate_k,
        )

        recommendations = []

        for _, row in candidates.iterrows():

            eligibility_text = row.get("eligibility", "")

            eligibility_result = self.eligibility_engine.evaluate(
                eligibility_text,
                user_profile,
            )

            if eligibility_result.status == "MISMATCH":
                continue

            result = row.copy()

            result["eligibility_status"] = (
                eligibility_result.status
            )

            result["eligibility_score"] = (
                eligibility_result.score
            )

            result["eligibility_checks"] = (
                eligibility_result.checks
            )

            recommendations.append(result)

        if not recommendations:
            return candidates.iloc[0:0].copy()

        result_df = candidates.iloc[0:0].copy()

        import pandas as pd

        result_df = pd.DataFrame(recommendations)

        # ---------------------------------------------------------
        # Sort: PRELIMINARY_MATCH first, then NEEDS_VERIFICATION,
        # then UNKNOWN.  Within each tier, higher TF-IDF similarity
        # score first (already present in the 'similarity' column
        # produced by SchemeTfidfEngine).
        # ---------------------------------------------------------
        _tier_order = {
            "PRELIMINARY_MATCH": 0,
            "NEEDS_VERIFICATION": 1,
            "UNKNOWN": 2,
        }
        result_df["_tier"] = result_df["eligibility_status"].map(
            lambda s: _tier_order.get(s, 9)
        )

        sim_col = "similarity_score" if "similarity_score" in result_df.columns else None

        if sim_col:
            result_df = result_df.sort_values(
                by=["_tier", sim_col],
                ascending=[True, False],
            )
        else:
            result_df = result_df.sort_values(
                by=["_tier"],
                ascending=[True],
            )

        result_df = result_df.drop(columns=["_tier"])

        return result_df.head(top_k).reset_index(drop=True)