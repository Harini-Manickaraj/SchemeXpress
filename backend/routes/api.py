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
    before making data requests. Also useful for container health checks
    and uptime monitors.

Adding more endpoints
---------------------
Keep each logical group in its own file, e.g.:
    backend/routes/recommendation_api.py  →  /api/recommend
    backend/routes/schemes_api.py         →  /api/schemes
Register each with app.register_blueprint() in app.py.
Never put business logic here — delegate to backend/services/.
"""

import os
import time

from flask import Blueprint, jsonify

# ------------------------------------------------------------------
# Blueprint definition
# url_prefix="/api" is applied at registration time in app.py.
# Naming the blueprint "api" keeps Flask's internal endpoint names
# predictable (e.g. url_for("api.health")).
# ------------------------------------------------------------------
api_bp = Blueprint("api", __name__)


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
