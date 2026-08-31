"""
backend/services/recommendation_service.py
===========================================
Recommendation service — owns the fitted SchemeRecommender instance.

DESIGN
------
Fitting TF-IDF on 3,397 schemes takes ~1-2 seconds and must not happen
on every HTTP request.  This module uses a module-level singleton:

    _recommender: SchemeRecommender | None = None

The first call to get_recommender() loads the dataset and fits the
recommender; subsequent calls return the already-fitted instance.

This is intentionally simple.  If the project later needs per-request
isolation (e.g. for A/B testing or hot-reload) this can be replaced
with a proper Flask application context extension.

The Flask routes in backend/routes/api.py call this module.
They do NOT import from src/ directly — all recommendation logic stays
in src/ and this service is the single gateway to it.

THREAD SAFETY
-------------
Flask's development server is single-threaded by default.  In
production (gunicorn/waitress) with multiple workers each worker
process has its own Python interpreter and its own copy of the module,
so the singleton is safe without a lock.  If a threaded single-worker
deployment is used, the initialization is safe because Python's GIL
protects the simple None-check assignment.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level singleton — None until first call to get_recommender()
_recommender = None


def get_recommender():
    """
    Return the fitted SchemeRecommender, initialising it on first call.

    Returns:
        SchemeRecommender: a fitted instance ready to call .recommend()

    Raises:
        FileNotFoundError: if the cleaned dataset is not found
        RuntimeError: if fitting fails for any reason
    """
    global _recommender

    if _recommender is not None:
        return _recommender

    # Import here so that importing this module (e.g. in tests that mock
    # the recommender) does not trigger heavy imports unless needed.
    from src.nlp.tfidf_engine import load_cleaned_schemes
    from src.recommendation.recommender import SchemeRecommender

    logger.info("SchemeRecommender: loading dataset…")
    try:
        df = load_cleaned_schemes()
    except FileNotFoundError:
        msg = (
            "Cleaned dataset not found. "
            "Run python run_phase2.py to generate it first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"SchemeRecommender: fitting on {len(df):,} schemes…")
    try:
        recommender = SchemeRecommender()
        recommender.fit(df)
    except Exception as exc:
        logger.exception("SchemeRecommender: fitting failed")
        raise RuntimeError(f"Failed to initialise recommender: {exc}") from exc

    _recommender = recommender
    logger.info("SchemeRecommender: ready.")
    return _recommender


def serialize_recommendations(results_df) -> list[dict]:
    """
    Convert the SchemeRecommender output DataFrame to a JSON-safe list.

    The DataFrame contains:
      - all original scheme columns (scheme_name, level, schemeCategory, etc.)
      - similarity_score      float
      - eligibility_status    str
      - eligibility_score     float
      - eligibility_checks    list[EligibilityCheck dataclasses]  ← not JSON-safe as-is

    We keep the most useful scheme fields and convert EligibilityCheck
    dataclasses to plain dicts.

    Fields included in each recommendation dict:
      scheme_name, slug, level, schemeCategory, tags,
      similarity_score, eligibility_status, eligibility_score,
      eligibility_checks (serialised)

    Fields intentionally excluded (too large for API response):
      details, benefits, eligibility, application, documents, combined_text
    These are available via a future GET /api/schemes/<slug> endpoint.
    """

    # Columns we want in the API response
    KEEP_COLS = [
        "scheme_name",
        "slug",
        "level",
        "schemeCategory",
        "tags",
        "similarity_score",
        "eligibility_status",
        "eligibility_score",
        "eligibility_checks",
    ]

    output = []

    for _, row in results_df.iterrows():
        rec: dict = {}

        # Plain scheme fields
        for col in KEEP_COLS:
            if col == "eligibility_checks":
                continue  # handled separately below
            if col in row.index:
                val = row[col]
                # Convert numpy floats/ints to Python natives for JSON
                if hasattr(val, "item"):
                    val = val.item()
                rec[col] = val

        # Serialise EligibilityCheck dataclasses
        checks_raw = row.get("eligibility_checks", [])
        rec["eligibility_checks"] = [
            {
                "criterion": chk.criterion,
                "status":    chk.status,
                "reason":    chk.reason,
            }
            for chk in (checks_raw or [])
        ]

        output.append(rec)

    return output
