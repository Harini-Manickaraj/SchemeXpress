"""
tests/test_api.py
==================
API tests for POST /api/recommend and GET /api/health.

Uses Flask's built-in test client — no pytest or extra packages needed.

Run from the project root:
    python tests/test_api.py

Tests:
    A. Valid scholarship request
    B. Valid construction-worker request
    C. Missing query
    D. Empty query
    E. Empty profile (valid — profile is optional)
    F. Invalid top_k (float, string, zero, negative, over-limit)
    G. Valid request — verify eligibility_checks are serialised correctly
    H. Invalid profile type (not a dict)
    I. Missing JSON body / wrong Content-Type
    J. Known MISMATCH scheme excluded from results
    K. GET /api/health still works after recommend addition
"""

import sys
import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ------------------------------------------------------------------
# Create the Flask test app and pre-warm the recommender.
# We patch the module-level singleton so fitting happens once,
# not once per test client instantiation.
# ------------------------------------------------------------------
from backend.app import create_app
import backend.services.recommendation_service as rec_svc

app = create_app()
app.testing = True

# Pre-warm the recommender now so individual tests run fast
print("Pre-warming recommender for API tests…", flush=True)
try:
    rec_svc.get_recommender()
    print("Recommender ready.\n", flush=True)
except Exception as exc:
    print(f"ERROR: could not initialise recommender: {exc}", file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------------
# Minimal assertion helper
# ------------------------------------------------------------------
PASS = 0
FAIL = 0


def _check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def _post(client, body: dict):
    """POST /api/recommend with JSON body."""
    return client.post(
        "/api/recommend",
        data=json.dumps(body),
        content_type="application/json",
    )


def _json(response) -> dict:
    return json.loads(response.data)


# ------------------------------------------------------------------
# Helper: check that a recommendation dict is well-formed
# ------------------------------------------------------------------
REQUIRED_FIELDS = {
    "scheme_name", "similarity_score", "eligibility_status",
    "eligibility_score", "eligibility_checks",
}

def _rec_is_complete(rec: dict) -> bool:
    return all(f in rec for f in REQUIRED_FIELDS)


def _checks_are_serialised(checks: list) -> bool:
    """Each eligibility check must be a plain dict, not a Python object."""
    if not isinstance(checks, list):
        return False
    for chk in checks:
        if not isinstance(chk, dict):
            return False
        if not all(k in chk for k in ("criterion", "status", "reason")):
            return False
    return True


# ==================================================================
# TEST A — Valid scholarship request
# ==================================================================
def test_a_valid_scholarship():
    print("\n─── Test A: valid scholarship request ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "scholarship for students",
            "profile": {
                "age": 16,
                "state": "Rajasthan",
                "is_student": True,
                "education_class": 8,
            },
            "top_k": 5,
        })
        data = _json(response)

    _check("HTTP 200", response.status_code == 200)
    _check("success=True", data.get("success") is True)
    _check("'recommendations' key present", "recommendations" in data)
    _check("'disclaimer' key present", "disclaimer" in data)
    _check("'count' key present", "count" in data)
    _check("count matches len(recommendations)",
           data.get("count") == len(data.get("recommendations", [])))
    _check("count > 0", data.get("count", 0) > 0)

    if data.get("recommendations"):
        rec = data["recommendations"][0]
        _check("scheme_name present",        "scheme_name"        in rec)
        _check("similarity_score present",   "similarity_score"   in rec)
        _check("eligibility_status present", "eligibility_status" in rec)
        _check("eligibility_score present",  "eligibility_score"  in rec)
        _check("eligibility_checks present", "eligibility_checks" in rec)
        _check("eligibility_checks is list", isinstance(rec.get("eligibility_checks"), list))
        _check("first rec is well-formed",   _rec_is_complete(rec))
        _check("eligibility_checks serialised correctly",
               _checks_are_serialised(rec.get("eligibility_checks", [])))
        _check("no MISMATCH in results",
               all(r.get("eligibility_status") != "MISMATCH"
                   for r in data["recommendations"]))

    print(f"  Top scheme: {data['recommendations'][0]['scheme_name'][:60]}"
          if data.get("recommendations") else "  (no results)")
    print(f"  Count: {data.get('count')}")


# ==================================================================
# TEST B — Valid construction-worker request
# ==================================================================
def test_b_valid_construction_worker():
    print("\n─── Test B: valid construction-worker request ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "assistance for construction workers",
            "profile": {
                "age": 25,
                "state": "Chhattisgarh",
                "occupation": "construction_worker",
                "is_registered": True,
            },
            "top_k": 5,
        })
        data = _json(response)

    _check("HTTP 200", response.status_code == 200)
    _check("success=True", data.get("success") is True)
    _check("count > 0", data.get("count", 0) > 0)
    _check("no MISMATCH in results",
           all(r.get("eligibility_status") != "MISMATCH"
               for r in data.get("recommendations", [])))

    print(f"  Count: {data.get('count')}")
    if data.get("recommendations"):
        print(f"  Top scheme: {data['recommendations'][0]['scheme_name'][:60]}")
        print(f"  Eligibility: {data['recommendations'][0]['eligibility_status']}")


# ==================================================================
# TEST C — Missing query field
# ==================================================================
def test_c_missing_query():
    print("\n─── Test C: missing query ───")
    with app.test_client() as client:
        response = _post(client, {
            "profile": {"age": 16},
        })
        data = _json(response)

    _check("HTTP 400", response.status_code == 400)
    _check("success=False", data.get("success") is False)
    _check("'error' key present", "error" in data)
    print(f"  Error message: {data.get('error')}")


# ==================================================================
# TEST D — Empty query string
# ==================================================================
def test_d_empty_query():
    print("\n─── Test D: empty query ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "   ",
            "profile": {},
        })
        data = _json(response)

    _check("HTTP 400", response.status_code == 400)
    _check("success=False", data.get("success") is False)
    _check("'error' key present", "error" in data)
    print(f"  Error message: {data.get('error')}")


# ==================================================================
# TEST E — Empty profile (profile is optional)
# ==================================================================
def test_e_empty_profile():
    print("\n─── Test E: empty profile (profile is optional) ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "scholarship for students",
            "profile": {},
            "top_k": 5,
        })
        data = _json(response)

    _check("HTTP 200 (empty profile is valid)",   response.status_code == 200)
    _check("success=True",                        data.get("success") is True)
    _check("count > 0",                           data.get("count", 0) > 0)
    _check("no MISMATCH in results",
           all(r.get("eligibility_status") != "MISMATCH"
               for r in data.get("recommendations", [])))

    # With empty profile, the engine cannot evaluate any criteria →
    # all results should be UNKNOWN (no criteria to match or mismatch)
    statuses = {r.get("eligibility_status") for r in data.get("recommendations", [])}
    print(f"  Statuses in results: {statuses}")
    print(f"  Count: {data.get('count')}")


# ==================================================================
# TEST F — Invalid top_k
# ==================================================================
def test_f_invalid_top_k():
    print("\n─── Test F: invalid top_k values ───")

    cases = [
        ("float",    {"query": "scholarship", "top_k": 5.5},    400),
        ("string",   {"query": "scholarship", "top_k": "ten"},  400),
        ("zero",     {"query": "scholarship", "top_k": 0},      400),
        ("negative", {"query": "scholarship", "top_k": -1},     400),
        ("over 50",  {"query": "scholarship", "top_k": 51},     400),
        ("bool",     {"query": "scholarship", "top_k": True},   400),  # bool is subclass of int
    ]

    with app.test_client() as client:
        for label, body, expected_status in cases:
            response = _post(client, body)
            data = _json(response)
            _check(
                f"top_k={body['top_k']!r} ({label}) → HTTP {expected_status}",
                response.status_code == expected_status,
            )
            if response.status_code == 400:
                _check(f"  └─ success=False", data.get("success") is False)


# ==================================================================
# TEST G — Eligibility checks are properly serialised
# ==================================================================
def test_g_eligibility_checks_serialised():
    print("\n─── Test G: eligibility_checks are plain dicts (not Python objects) ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "scholarship engineering students Chhattisgarh",
            "profile": {
                "state": "Chhattisgarh",
                "is_student": True,
                "course": "B.Tech",
                "study_year": 1,
                "class_12_percentage": 72,
                "receives_other_scholarship": False,
            },
            "top_k": 5,
        })
        data = _json(response)

    _check("HTTP 200", response.status_code == 200)
    _check("success=True", data.get("success") is True)

    all_recs = data.get("recommendations", [])
    if all_recs:
        for rec in all_recs:
            checks = rec.get("eligibility_checks", [])
            _check(
                f"  checks for '{rec.get('scheme_name','?')[:45]}' are dicts",
                _checks_are_serialised(checks),
            )

    # Verify the CG scheme if it appears
    cg_name = (
        "Scholarship for C.G. Domicile Students Pursuing Engineering Courses "
        "Outside Chhattisgarh State"
    )
    cg_hits = [r for r in all_recs if r.get("scheme_name") == cg_name]
    if cg_hits:
        cg = cg_hits[0]
        _check("CG scheme has PRELIMINARY_MATCH",
               cg.get("eligibility_status") == "PRELIMINARY_MATCH")
        cg_checks = cg.get("eligibility_checks", [])
        _check("CG checks: at least one MATCH criterion",
               any(c.get("status") == "MATCH" for c in cg_checks))
        print(f"  CG scheme found at eligibility: {cg.get('eligibility_status')}")
        print(f"  CG checks: {[(c['criterion'], c['status']) for c in cg_checks]}")
    else:
        print(f"  CG scheme not in top-5 for this query (acceptable)")


# ==================================================================
# TEST H — Invalid profile type
# ==================================================================
def test_h_invalid_profile_type():
    print("\n─── Test H: invalid profile type (string instead of dict) ───")
    with app.test_client() as client:
        response = _post(client, {
            "query": "scholarship for students",
            "profile": "age=16,state=Rajasthan",
        })
        data = _json(response)

    _check("HTTP 400", response.status_code == 400)
    _check("success=False", data.get("success") is False)
    _check("'error' key present", "error" in data)
    print(f"  Error: {data.get('error')}")


# ==================================================================
# TEST I — Missing JSON body / wrong Content-Type
# ==================================================================
def test_i_missing_body():
    print("\n─── Test I: non-JSON body ───")
    with app.test_client() as client:
        # Send plain text, no JSON content-type
        response = client.post(
            "/api/recommend",
            data="not json at all",
            content_type="text/plain",
        )
        data = _json(response)

    _check("HTTP 400", response.status_code == 400)
    _check("success=False", data.get("success") is False)
    print(f"  Error: {data.get('error')}")


# ==================================================================
# TEST J — MISMATCH scheme is excluded (invalid CG profile)
# ==================================================================
def test_j_mismatch_excluded():
    print("\n─── Test J: MISMATCH scheme excluded for invalid profile ───")
    cg_name = (
        "Scholarship for C.G. Domicile Students Pursuing Engineering Courses "
        "Outside Chhattisgarh State"
    )
    with app.test_client() as client:
        response = _post(client, {
            "query": "scholarship engineering students Chhattisgarh",
            "profile": {
                "state": "Chhattisgarh",
                "is_student": True,
                "course": "B.Tech",
                "study_year": 2,             # MISMATCH: scheme requires 1st year
                "class_12_percentage": 55,   # MISMATCH: below 60%
                "receives_other_scholarship": True,  # MISMATCH: not allowed
            },
            "top_k": 10,
        })
        data = _json(response)

    _check("HTTP 200", response.status_code == 200)
    _check("no MISMATCH in results",
           all(r.get("eligibility_status") != "MISMATCH"
               for r in data.get("recommendations", [])))
    names = [r.get("scheme_name") for r in data.get("recommendations", [])]
    _check("CG engineering scholarship absent from results",
           cg_name not in names)

    print(f"  Results count: {data.get('count', 0)}")
    print(f"  CG scheme present: {cg_name in names}")


# ==================================================================
# TEST K — GET /api/health still works
# ==================================================================
def test_k_health_still_works():
    print("\n─── Test K: GET /api/health unaffected ───")
    with app.test_client() as client:
        response = client.get("/api/health")
        data = _json(response)

    _check("HTTP 200",       response.status_code == 200)
    _check("status=ok",      data.get("status") == "ok")
    _check("service present", "service" in data)
    _check("timestamp present", "timestamp" in data)
    print(f"  Health: {data}")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("SchemeXpress — POST /api/recommend API Test Suite")
    print("=" * 65)

    test_a_valid_scholarship()
    test_b_valid_construction_worker()
    test_c_missing_query()
    test_d_empty_query()
    test_e_empty_profile()
    test_f_invalid_top_k()
    test_g_eligibility_checks_serialised()
    test_h_invalid_profile_type()
    test_i_missing_body()
    test_j_mismatch_excluded()
    test_k_health_still_works()

    print("\n" + "=" * 65)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("ALL API TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — review output above")
        sys.exit(1)
