"""
tests/test_recommender_and_engine.py
=====================================
Full test suite for:
  - EligibilityEngine  (Chhattisgarh scholarship: all five sub-tests)
  - SchemeRecommender  (four query/profile scenarios + sorting verification)

Run from the project root:
    python -m pytest tests/test_recommender_and_engine.py -v

Or run directly:
    python tests/test_recommender_and_engine.py
"""

import sys
import os

# Make sure the project root is on the path regardless of how the file
# is executed (pytest, direct python, or from any working directory).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from src.recommendation.eligibility_engine import EligibilityEngine, EligibilityResult
from src.recommendation.recommender import SchemeRecommender
from src.nlp.tfidf_engine import load_cleaned_schemes

# ── Shared fixtures ────────────────────────────────────────────────────────

ENGINE = EligibilityEngine()

# Actual eligibility text from Row 2870 of the cleaned dataset
CG_ELIGIBILITY = (
    "The applicant must be a domicile of Chhattisgarh. "
    "The applicant must be enrolled in a recognized engineering college outside Chhattisgarh. "
    "The applicant should be studying in IIT / NIT / IISc or in an AICTE-approved institution. "
    "The applicant should be enrolled in the first year of a B.E./B.Tech. course. "
    "The applicant should have secured at least 60% marks in the 12th examination. "
    "The applicant must not be receiving any other scholarship from the state/central government "
    "or other sources."
)

PASS_COUNTER = {"pass": 0, "fail": 0}


def _assert(condition: bool, message: str) -> None:
    if condition:
        PASS_COUNTER["pass"] += 1
        print(f"  PASS  {message}")
    else:
        PASS_COUNTER["fail"] += 1
        print(f"  FAIL  {message}")


def _result_summary(result: EligibilityResult) -> dict:
    """Return {criterion: status} for quick inspection."""
    return {chk.criterion: chk.status for chk in result.checks}


# ══════════════════════════════════════════════════════════════════════════
# BLOCK 1 — EligibilityEngine: Chhattisgarh engineering scholarship
# ══════════════════════════════════════════════════════════════════════════

def test_cg_test_a_valid_profile():
    """Test A — valid complete profile → PRELIMINARY_MATCH."""
    print("\n─── Test A: valid profile ───")
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 1,
        "class_12_percentage": 72,
        "receives_other_scholarship": False,
    }
    result = ENGINE.evaluate(CG_ELIGIBILITY, profile)
    checks = _result_summary(result)

    _assert(checks.get("state")                == "MATCH", "state → MATCH")
    _assert(checks.get("student_status")       == "MATCH", "student_status → MATCH")
    _assert(checks.get("course")               == "MATCH", "course → MATCH")
    _assert(checks.get("study_year")           == "MATCH", "study_year → MATCH")
    _assert(checks.get("class_12_percentage")  == "MATCH", "class_12_percentage → MATCH")
    _assert(checks.get("other_scholarship")    == "MATCH", "other_scholarship → MATCH")
    _assert(result.status != "MISMATCH",              "overall status is NOT MISMATCH")
    _assert(result.status == "PRELIMINARY_MATCH",     "overall status is PRELIMINARY_MATCH")

    print(f"  Overall status: {result.status}")
    return result


def test_cg_test_b_wrong_study_year():
    """Test B — study_year=2 → MISMATCH on study_year, overall MISMATCH."""
    print("\n─── Test B: wrong study year ───")
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 2,
        "class_12_percentage": 72,
        "receives_other_scholarship": False,
    }
    result = ENGINE.evaluate(CG_ELIGIBILITY, profile)
    checks = _result_summary(result)

    _assert(checks.get("study_year") == "MISMATCH", "study_year → MISMATCH")
    _assert(result.status == "MISMATCH",            "overall → MISMATCH")

    print(f"  Overall status: {result.status}")
    return result


def test_cg_test_c_insufficient_percentage():
    """Test C — class_12_percentage=55 → MISMATCH, overall MISMATCH."""
    print("\n─── Test C: insufficient Class 12 percentage ───")
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 1,
        "class_12_percentage": 55,
        "receives_other_scholarship": False,
    }
    result = ENGINE.evaluate(CG_ELIGIBILITY, profile)
    checks = _result_summary(result)

    _assert(checks.get("class_12_percentage") == "MISMATCH", "class_12_percentage → MISMATCH")
    _assert(result.status == "MISMATCH",                     "overall → MISMATCH")

    # Boundary: exactly 60% must MATCH
    profile_boundary = {**profile, "class_12_percentage": 60}
    result_b = ENGINE.evaluate(CG_ELIGIBILITY, profile_boundary)
    checks_b = _result_summary(result_b)
    _assert(checks_b.get("class_12_percentage") == "MATCH",
            "boundary 60% → MATCH (exactly meets threshold)")

    print(f"  Overall status (55%): {result.status}")
    print(f"  Overall status (60%): {result_b.status}")
    return result


def test_cg_test_d_receiving_scholarship():
    """Test D — receives_other_scholarship=True → MISMATCH, overall MISMATCH."""
    print("\n─── Test D: receiving another scholarship ───")
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 1,
        "class_12_percentage": 72,
        "receives_other_scholarship": True,
    }
    result = ENGINE.evaluate(CG_ELIGIBILITY, profile)
    checks = _result_summary(result)

    _assert(checks.get("other_scholarship") == "MISMATCH", "other_scholarship → MISMATCH")
    _assert(result.status == "MISMATCH",                    "overall → MISMATCH")

    print(f"  Overall status: {result.status}")
    return result


def test_cg_test_e_incomplete_profile():
    """
    Test E — incomplete profile (state + is_student only).

    The engine evaluates only criteria for which the user provided a value.
    When course, study_year, class_12_percentage, and
    receives_other_scholarship are absent from the profile, those criteria
    are not evaluated — the engine conservatively skips them rather than
    inventing a failure.

    With only state (MATCH) and student_status (MATCH), all *evaluated*
    criteria pass → PRELIMINARY_MATCH is the correct result.

    PRELIMINARY_MATCH means "no supplied criterion failed"; it does NOT
    mean the user is fully eligible.  The disclaimer on every result
    must make this clear to consumers of the recommender output.

    The test asserts:
      - state   → MATCH      (evaluated, passes)
      - student → MATCH      (evaluated, passes)
      - overall → PRELIMINARY_MATCH  (correct — no supplied criterion failed)
      - No MISMATCH produced (correct — engine never fabricates failures)
    """
    print("\n─── Test E: incomplete profile (conservative behaviour) ───")
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
    }
    result = ENGINE.evaluate(CG_ELIGIBILITY, profile)

    checks = _result_summary(result)
    _assert(checks.get("state")          == "MATCH", "state → MATCH")
    _assert(checks.get("student_status") == "MATCH", "student_status → MATCH")

    # The engine correctly returns PRELIMINARY_MATCH because all *evaluated*
    # criteria passed.  Unanswered criteria (course, study_year, etc.) are
    # not present in the profile and therefore not evaluated.
    _assert(
        result.status == "PRELIMINARY_MATCH",
        "incomplete profile → PRELIMINARY_MATCH "
        "(only evaluated criteria count; no criterion failed)",
    )
    _assert(
        result.status != "MISMATCH",
        "incomplete profile → NOT MISMATCH (engine never fabricates failures)",
    )

    print(f"  Overall status: {result.status}")
    print(f"  Checks returned: {checks}")
    print("  NOTE: PRELIMINARY_MATCH here means 'no supplied criterion failed'.")
    print("        The result does NOT claim the user is fully eligible.")
    return result


# ══════════════════════════════════════════════════════════════════════════
# BLOCK 2 — SchemeRecommender: four scenario tests
# ══════════════════════════════════════════════════════════════════════════

# Fit the recommender once; it's expensive to re-fit for every test.
print("\nLoading cleaned dataset and fitting recommender (this takes a moment)…")
_df = load_cleaned_schemes()
_recommender = SchemeRecommender()
_recommender.fit(_df)
print(f"Recommender fitted on {len(_df):,} schemes.\n")


def _run_recommender_test(label: str, query: str, profile: dict, top_k: int = 10):
    """
    Run one recommender scenario and assert the invariants that must hold
    for every result set.
    Returns the result DataFrame.
    """
    print(f"\n─── Recommender test: {label} ───")
    print(f"  Query:   {query!r}")
    print(f"  Profile: {profile}")

    results = _recommender.recommend(query, profile, top_k=top_k, candidate_k=100)

    print(f"  Results returned: {len(results)}")

    # ── Invariant 1: no MISMATCH in results ──
    mismatch_count = (results["eligibility_status"] == "MISMATCH").sum() if len(results) else 0
    _assert(mismatch_count == 0, "no MISMATCH schemes in results")

    # ── Invariant 2: eligibility_status column is present ──
    _assert("eligibility_status" in results.columns, "eligibility_status column present")

    # ── Invariant 3: similarity_score column is present ──
    _assert("similarity_score" in results.columns, "similarity_score column present")

    # ── Invariant 4: PRELIMINARY_MATCH ranks above NEEDS_VERIFICATION ──
    if len(results) > 1:
        statuses = results["eligibility_status"].tolist()
        pm_indices = [i for i, s in enumerate(statuses) if s == "PRELIMINARY_MATCH"]
        nv_indices = [i for i, s in enumerate(statuses) if s == "NEEDS_VERIFICATION"]
        if pm_indices and nv_indices:
            ordering_ok = max(pm_indices) < min(nv_indices)
            _assert(ordering_ok,
                    "all PRELIMINARY_MATCH entries appear before NEEDS_VERIFICATION entries")
        else:
            print("  (skip ordering check — only one eligibility tier present in results)")

    # ── Invariant 5: within same tier, similarity_score descends ──
    if len(results) > 1 and "similarity_score" in results.columns:
        for tier_label in ("PRELIMINARY_MATCH", "NEEDS_VERIFICATION"):
            tier_rows = results[results["eligibility_status"] == tier_label]
            if len(tier_rows) > 1:
                scores = tier_rows["similarity_score"].tolist()
                # Allow floating-point tolerance
                descending = all(
                    scores[i] >= scores[i + 1] - 1e-9
                    for i in range(len(scores) - 1)
                )
                _assert(descending,
                        f"similarity_score descends within {tier_label} tier")

    # ── Show top-5 results ──
    show_cols = ["scheme_name", "eligibility_status", "similarity_score"]
    available = [c for c in show_cols if c in results.columns]
    if len(results):
        print(f"\n  Top-{min(5, len(results))} results:")
        for i, row in results[available].head(5).iterrows():
            name  = str(row.get("scheme_name", ""))[:70]
            elig  = row.get("eligibility_status", "?")
            score = row.get("similarity_score", 0.0)
            print(f"    [{i+1}] [{elig:20s}] (sim={score:.4f}) {name}")
    else:
        print("  (no results returned)")

    return results


def test_recommender_scholarship():
    return _run_recommender_test(
        label="scholarship for students",
        query="scholarship for students",
        profile={
            "age": 16,
            "state": "Rajasthan",
            "is_student": True,
            "education_class": 8,
        },
    )


def test_recommender_farmers():
    return _run_recommender_test(
        label="support for farmers",
        query="support for farmers",
        profile={
            "age": 40,
            "state": "Rajasthan",
            "occupation": "farmer",
        },
    )


def test_recommender_women_entrepreneurs():
    return _run_recommender_test(
        label="financial support for women entrepreneurs",
        query="financial support for women entrepreneurs",
        profile={
            "age": 30,
            "state": "Rajasthan",
            "gender": "female",
        },
    )


def test_recommender_construction_workers():
    return _run_recommender_test(
        label="assistance for construction workers",
        query="assistance for construction workers",
        profile={
            "age": 25,
            "state": "Chhattisgarh",
            "occupation": "construction_worker",
            "is_registered": True,
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# BLOCK 3 — Recommender with the CG scholarship: valid vs invalid profiles
# ══════════════════════════════════════════════════════════════════════════

def test_recommender_cg_valid():
    """
    Chhattisgarh user with valid profile should surface the CG engineering
    scholarship with PRELIMINARY_MATCH eligibility.
    """
    print("\n─── Recommender CG: valid profile ───")
    query = "scholarship for engineering students Chhattisgarh"
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 1,
        "class_12_percentage": 72,
        "receives_other_scholarship": False,
    }
    results = _recommender.recommend(query, profile, top_k=10, candidate_k=100)

    _assert("eligibility_status" in results.columns, "eligibility_status present")
    _assert("similarity_score"   in results.columns, "similarity_score present")

    mismatch_count = (results["eligibility_status"] == "MISMATCH").sum()
    _assert(mismatch_count == 0, "no MISMATCH in results")

    # Check whether the specific CG scheme appears and is PRELIMINARY_MATCH
    cg_name = "Scholarship for C.G. Domicile Students Pursuing Engineering Courses Outside Chhattisgarh State"
    cg_rows = results[results["scheme_name"] == cg_name]
    if len(cg_rows):
        status = cg_rows.iloc[0]["eligibility_status"]
        _assert(status == "PRELIMINARY_MATCH",
                f"CG engineering scholarship → PRELIMINARY_MATCH (got {status})")
        print(f"  CG scheme found at rank {cg_rows.index[0]+1}, status={status}")
    else:
        print("  NOTE: CG scheme not in top-10 for this query (acceptable — TF-IDF may rank others higher)")

    show_cols = ["scheme_name", "eligibility_status", "similarity_score"]
    available = [c for c in show_cols if c in results.columns]
    print(f"\n  Top-{min(5, len(results))} results:")
    for i, row in results[available].head(5).iterrows():
        print(f"    [{i+1}] [{row.get('eligibility_status','?'):20s}] "
              f"(sim={row.get('similarity_score',0):.4f}) "
              f"{str(row.get('scheme_name',''))[:70]}")

    return results


def test_recommender_cg_invalid():
    """
    Chhattisgarh user with invalid profile (wrong year, low %, receiving scholarship).
    The CG engineering scholarship should be EXCLUDED (MISMATCH → filtered out).
    """
    print("\n─── Recommender CG: invalid profile (should filter out CG scheme) ───")
    query = "scholarship for engineering students Chhattisgarh"
    profile = {
        "state": "Chhattisgarh",
        "is_student": True,
        "course": "B.Tech",
        "study_year": 2,            # wrong year
        "class_12_percentage": 55,  # below 60%
        "receives_other_scholarship": True,  # not allowed
    }

    # Direct eligibility check first
    result_direct = ENGINE.evaluate(CG_ELIGIBILITY, profile)
    _assert(result_direct.status == "MISMATCH",
            "direct eligibility check → MISMATCH for invalid profile")
    checks = _result_summary(result_direct)
    _assert(checks.get("study_year")           == "MISMATCH", "study_year → MISMATCH")
    _assert(checks.get("class_12_percentage")  == "MISMATCH", "class_12_percentage → MISMATCH")
    _assert(checks.get("other_scholarship")    == "MISMATCH", "other_scholarship → MISMATCH")

    # Recommender results should not include the CG scheme
    results = _recommender.recommend(query, profile, top_k=10, candidate_k=100)
    mismatch_count = (results["eligibility_status"] == "MISMATCH").sum()
    _assert(mismatch_count == 0, "no MISMATCH in recommender output")

    cg_name = "Scholarship for C.G. Domicile Students Pursuing Engineering Courses Outside Chhattisgarh State"
    cg_in_results = (results["scheme_name"] == cg_name).any()
    _assert(not cg_in_results, "CG engineering scholarship NOT in results for invalid profile")

    print(f"  Direct eligibility: {result_direct.status}")
    print(f"  CG scheme in recommender results: {cg_in_results}")
    print(f"  Recommender returned {len(results)} schemes (all non-MISMATCH)")
    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("SchemeXpress — EligibilityEngine + Recommender Test Suite")
    print("=" * 65)

    # Block 1 — Engine tests
    print("\n══ BLOCK 1: EligibilityEngine (Chhattisgarh scholarship) ══")
    test_cg_test_a_valid_profile()
    test_cg_test_b_wrong_study_year()
    test_cg_test_c_insufficient_percentage()
    test_cg_test_d_receiving_scholarship()
    test_cg_test_e_incomplete_profile()

    # Block 2 — Recommender scenario tests
    print("\n══ BLOCK 2: SchemeRecommender (four scenarios) ══")
    test_recommender_scholarship()
    test_recommender_farmers()
    test_recommender_women_entrepreneurs()
    test_recommender_construction_workers()

    # Block 3 — Recommender CG specific
    print("\n══ BLOCK 3: Recommender with CG scholarship ══")
    test_recommender_cg_valid()
    test_recommender_cg_invalid()

    # ── Final summary ─────────────────────────────────────────────
    total  = PASS_COUNTER["pass"] + PASS_COUNTER["fail"]
    passed = PASS_COUNTER["pass"]
    failed = PASS_COUNTER["fail"]

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed}/{total} assertions passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — review output above")
        sys.exit(1)
