"""
backend/routes/api.py
=====================
SchemeXpress — Public JSON API routes.

All routes in this file are served under the /api prefix
(set when the blueprint is registered in app.py).

Current endpoints
-----------------
GET /api/health
    Lightweight liveness probe.
    Used by the Scheme Genie frontend to confirm the backend is reachable
    before making data requests.

POST /api/recommend
    Hybrid recommendation endpoint.
    Accepts a user query + profile, runs the SchemeRecommender pipeline,
    and returns a ranked list of government schemes.

Adding more endpoints
---------------------
Keep each logical group in its own file, e.g.:
    backend/routes/schemes_api.py  →  /api/schemes/<slug>
Register each with app.register_blueprint() in app.py.
Never put recommendation logic directly in routes — delegate to
backend/services/recommendation_service.py.
"""

import os
import time
import logging

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Blueprint definition
# url_prefix="/api" is applied at registration time in app.py.
# ------------------------------------------------------------------
api_bp = Blueprint("api", __name__)

# Disclaimer included in every recommendation response.
# Ensures consumers cannot mistake preliminary results for legal decisions.
_DISCLAIMER = (
    "Eligibility results are preliminary and should be verified against "
    "the official scheme guidelines. This system does not provide legally "
    "verified eligibility decisions."
)


@api_bp.route("/health", methods=["GET"])
def health():
    """
    GET /api/health

    Returns a JSON object confirming the backend is alive.

    Response shape:
    {
        "status":    "ok",
        "service":   "SchemeXpress",
        "version":   "0.1.0",
        "timestamp": <unix epoch float>,
        "env":       "development" | "production"
    }

    HTTP 200 always (a non-200 means Flask itself failed to start).
    """
    return jsonify({
        "status":    "ok",
        "service":   "SchemeXpress",
        "version":   "0.1.0",
        "timestamp": time.time(),
        "env":       os.getenv("FLASK_ENV", "development"),
    }), 200


@api_bp.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /api/recommend

    Accepts a JSON body and returns a ranked list of government schemes.

    Request body:
    {
        "query":   "scholarship for students",   // required, non-empty string
        "profile": {                              // optional dict
            "age": 16,
            "state": "Rajasthan",
            "is_student": true,
            "education_class": 8
        },
        "top_k": 10                              // optional int, 1-50, default 10
    }

    Success response (HTTP 200):
    {
        "success": true,
        "disclaimer": "...",
        "count": <int>,
        "recommendations": [
            {
                "scheme_name": "...",
                "slug": "...",
                "level": "...",
                "schemeCategory": "...",
                "tags": "...",
                "similarity_score": 0.30,
                "eligibility_status": "PRELIMINARY_MATCH",
                "eligibility_score": 1.0,
                "eligibility_checks": [
                    {"criterion": "state", "status": "MATCH", "reason": "..."},
                    ...
                ]
            }
        ]
    }

    Error responses:
        400 — missing/empty query, invalid types, invalid top_k
        500 — internal recommendation error
    """
    # ----------------------------------------------------------
    # 1. Parse request body
    # ----------------------------------------------------------
    body = request.get_json(silent=True)  # silent=True → None instead of 400

    if body is None:
        return jsonify({
            "success": False,
            "error":   "Request body must be valid JSON with Content-Type: application/json",
        }), 400

    # ----------------------------------------------------------
    # 2. Validate query
    # ----------------------------------------------------------
    query = body.get("query")

    if query is None:
        return jsonify({
            "success": False,
            "error":   "'query' is required.",
        }), 400

    if not isinstance(query, str):
        return jsonify({
            "success": False,
            "error":   "'query' must be a string.",
        }), 400

    query = query.strip()

    if not query:
        return jsonify({
            "success": False,
            "error":   "'query' must not be empty.",
        }), 400

    # ----------------------------------------------------------
    # 3. Validate profile (optional)
    # ----------------------------------------------------------
    profile = body.get("profile", {})

    if profile is None:
        profile = {}

    if not isinstance(profile, dict):
        return jsonify({
            "success": False,
            "error":   "'profile' must be a JSON object (dictionary).",
        }), 400

    # ----------------------------------------------------------
    # 4. Validate top_k (optional, default 10, max 50)
    # ----------------------------------------------------------
    TOP_K_DEFAULT = 10
    TOP_K_MAX     = 50

    raw_top_k = body.get("top_k", TOP_K_DEFAULT)

    if not isinstance(raw_top_k, int) or isinstance(raw_top_k, bool):
        return jsonify({
            "success": False,
            "error":   f"'top_k' must be an integer (received {type(raw_top_k).__name__}).",
        }), 400

    if raw_top_k < 1 or raw_top_k > TOP_K_MAX:
        return jsonify({
            "success": False,
            "error":   f"'top_k' must be between 1 and {TOP_K_MAX}.",
        }), 400

    top_k = raw_top_k

    # ----------------------------------------------------------
    # 5. Get the fitted recommender (initialises on first call)
    # ----------------------------------------------------------
    try:
        from backend.services.recommendation_service import (
            get_recommender,
            serialize_recommendations,
        )
        recommender = get_recommender()
    except FileNotFoundError as exc:
        logger.error(f"Dataset not found: {exc}")
        return jsonify({
            "success": False,
            "error":   "Recommendation service unavailable: dataset not found.",
        }), 500
    except RuntimeError as exc:
        logger.error(f"Recommender init failed: {exc}")
        return jsonify({
            "success": False,
            "error":   "Recommendation service unavailable: initialisation failed.",
        }), 500

    # ----------------------------------------------------------
    # 6. Run recommendation
    # ----------------------------------------------------------
    try:
        results_df = recommender.recommend(
            query,
            profile,
            top_k=top_k,
        )
    except ValueError as exc:
        # e.g. empty query slipping through (defensive)
        return jsonify({
            "success": False,
            "error":   str(exc),
        }), 400
    except Exception as exc:
        logger.exception("Recommendation failed")
        return jsonify({
            "success": False,
            "error":   "An internal error occurred during recommendation.",
        }), 500

    # ----------------------------------------------------------
    # 7. Serialise and return
    # ----------------------------------------------------------
    try:
        recommendations = serialize_recommendations(results_df)
    except Exception as exc:
        logger.exception("Serialisation failed")
        return jsonify({
            "success": False,
            "error":   "An internal error occurred while formatting results.",
        }), 500

    return jsonify({
        "success":         True,
        "disclaimer":      _DISCLAIMER,
        "count":           len(recommendations),
        "recommendations": recommendations,
    }), 200
