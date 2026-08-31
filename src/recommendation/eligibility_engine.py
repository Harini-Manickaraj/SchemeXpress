# src/recommendation/eligibility_engine.py

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EligibilityCheck:
    criterion: str
    status: str
    reason: str


@dataclass
class EligibilityResult:
    status: str
    checks: list[EligibilityCheck]

    @property
    def score(self) -> float:
        """
        Score only the criteria that could actually be evaluated.

        This is NOT an eligibility percentage.
        """
        evaluated = [
            check for check in self.checks
            if check.status in {"MATCH", "MISMATCH"}
        ]

        if not evaluated:
            return 0.0

        matches = sum(
            check.status == "MATCH"
            for check in evaluated
        )

        return matches / len(evaluated)


class EligibilityEngine:
    """
    Conservative rule-based eligibility evaluator.

    Supported criteria:
        - age
        - annual income
        - state/residence
        - gender
        - SC/ST/OBC category
        - occupation
        - student status
        - education class
        - income-taxpayer status
        - land ownership
        - registration

    The engine does not claim legal eligibility.
    Unsupported or missing information remains UNKNOWN.
    """

    def evaluate(
        self,
        eligibility_text: str,
        user_profile: dict[str, Any],
    ) -> EligibilityResult:

        if not eligibility_text or not str(eligibility_text).strip():
            return EligibilityResult(
                status="UNKNOWN",
                checks=[
                    EligibilityCheck(
                        criterion="general",
                        status="UNKNOWN",
                        reason="No eligibility information available.",
                    )
                ],
            )

        text = str(eligibility_text).lower()

        checks = []

        for check in [
            self._check_age(text, user_profile),
            self._check_income(text, user_profile),
            self._check_state(text, user_profile),
            self._check_gender(text, user_profile),
            self._check_category(text, user_profile),
            self._check_occupation(text, user_profile),
            self._check_student(text, user_profile),
            self._check_education_class(text, user_profile),
            self._check_taxpayer(text, user_profile),
            self._check_land(text, user_profile),
            self._check_registration(text, user_profile),
            self._check_course(text, user_profile),
	    self._check_study_year(text, user_profile),
	    self._check_class_12_percentage(text, user_profile),
	    self._check_other_scholarship(text, user_profile),
        ]:
            if check is not None:
                checks.append(check)

        if not checks:
            return EligibilityResult(
                status="UNKNOWN",
                checks=[
                    EligibilityCheck(
                        criterion="general",
                        status="UNKNOWN",
                        reason=(
                            "No supported eligibility requirement "
                            "could be evaluated."
                        ),
                    )
                ],
            )

        if any(check.status == "MISMATCH" for check in checks):
            overall_status = "MISMATCH"
        elif any(check.status == "UNKNOWN" for check in checks):
            overall_status = "NEEDS_VERIFICATION"
        else:
            overall_status = "PRELIMINARY_MATCH"

        return EligibilityResult(
            status=overall_status,
            checks=checks,
        )

    # ---------------------------------------------------------
    # AGE
    # ---------------------------------------------------------

    def _check_age(self, text, profile):

        age = profile.get("age")

        if age is None:
            return None

        patterns = [
            r"age(?: group)?\s*(?:of)?\s*(\d+)\s*[-–]\s*(\d+)",
            r"aged?\s*(\d+)\s*[-–]\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                minimum = int(match.group(1))
                maximum = int(match.group(2))

                if minimum <= age <= maximum:
                    return EligibilityCheck(
                        "age",
                        "MATCH",
                        f"Age {age} is within {minimum}-{maximum}.",
                    )

                return EligibilityCheck(
                    "age",
                    "MISMATCH",
                    f"Age {age} is outside {minimum}-{maximum}.",
                )

        minimum_patterns = [
    		r"age.*?(\d+)\s*years?\s*(?:or more|and above)",
    		r"age.*?not less than\s*(\d+)",
    		r"age.*?should not be less than\s*(\d+)",
    		r"age.*?minimum.*?(\d+)",
	]

        for pattern in minimum_patterns:
            match = re.search(pattern, text)

            if match:
                minimum = int(match.group(1))

                if age >= minimum:
                    return EligibilityCheck(
                        "age",
                        "MATCH",
                        f"Age {age} meets minimum age {minimum}.",
                    )

                return EligibilityCheck(
                    "age",
                    "MISMATCH",
                    f"Age {age} is below minimum age {minimum}.",
                )

        return None

    # ---------------------------------------------------------
    # INCOME
    # ---------------------------------------------------------

    def _check_income(self, text, profile):

        income = profile.get("annual_income")

        if income is None:
            return None

        amounts = re.findall(
            r"(?:₹|rs\.?|rs)\s*([\d,]+)",
            text,
        )

        if not amounts:
            return None

        values = [
            int(value.replace(",", ""))
            for value in amounts
        ]

        if any(
            phrase in text
            for phrase in [
                "not exceed",
                "less than",
                "below",
                "does not exceed",
                "maximum income",
            ]
        ):
            limit = min(values)

            if income <= limit:
                return EligibilityCheck(
                    "annual_income",
                    "MATCH",
                    f"Annual income ₹{income:,} is within ₹{limit:,}.",
                )

            return EligibilityCheck(
                "annual_income",
                "MISMATCH",
                f"Annual income ₹{income:,} exceeds ₹{limit:,}.",
            )

        return None

    # ---------------------------------------------------------
    # STATE / RESIDENCE
    # ---------------------------------------------------------

    def _check_state(self, text, profile):

        state = profile.get("state")

        if not state:
            return None

        residence_terms = [
           "resident",
    	   "reside",
           "native of",
   	   "permanent resident",
  	   "belong to",
	   "domicile",
        ]

        if not any(term in text for term in residence_terms):
            return None

        state_normalized = state.lower()

        if state_normalized in text:
            return EligibilityCheck(
                "state",
                "MATCH",
                f"Eligibility mentions {state}.",
            )

        return EligibilityCheck(
            "state",
            "MISMATCH",
            f"Eligibility does not mention {state}.",
        )

    # ---------------------------------------------------------
    # GENDER
    # ---------------------------------------------------------

    def _check_gender(self, text, profile):

        gender = profile.get("gender")

        if not gender:
            return None

        female_terms = [
            "woman",
            "women",
            "female",
            "girl",
            "girls",
            "mahila",
        ]

        if any(term in text for term in female_terms):

            if gender.lower() in {
                "female",
                "woman",
                "girl",
            }:
                return EligibilityCheck(
                    "gender",
                    "MATCH",
                    "Scheme contains a female/women requirement.",
                )

            return EligibilityCheck(
                "gender",
                "MISMATCH",
                "Scheme contains a female/women requirement.",
            )

        return None

    # ---------------------------------------------------------
    # CATEGORY
    # ---------------------------------------------------------

    def _check_category(self, text, profile):

        category = profile.get("category")

        if not category:
            return None

        category = category.lower()

        category_terms = {
            "sc": [
                "scheduled caste",
                "sc community",
            ],
            "st": [
                "scheduled tribe",
                "st community",
            ],
            "obc": [
                "obc",
                "other backward class",
            ],
        }

        terms = category_terms.get(category)

        if not terms:
            return None

        if any(term in text for term in terms):
            return EligibilityCheck(
                "category",
                "MATCH",
                f"Scheme contains a {category.upper()} requirement.",
            )

        return None

    # ---------------------------------------------------------
    # OCCUPATION
    # ---------------------------------------------------------

    def _check_occupation(self, text, profile):

        occupation = profile.get("occupation")

        if not occupation:
            return None

        occupation = occupation.lower()

        occupation_terms = {
            "farmer": [
                "farmer",
                "farmers",
                "agriculture",
            ],
            "construction_worker": [
                "construction worker",
                "construction workers",
            ],
            "student": [
                "student",
                "students",
            ],
            "fisherman": [
                "fisherman",
                "fishermen",
                "fisherwoman",
            ],
        }

        terms = occupation_terms.get(occupation)

        if not terms:
            return None

        if any(term in text for term in terms):
            return EligibilityCheck(
                "occupation",
                "MATCH",
                f"Scheme is related to {occupation.replace('_', ' ')}.",
            )

        return None

    # ---------------------------------------------------------
    # STUDENT STATUS
    # ---------------------------------------------------------

    def _check_student(self, text, profile):

        is_student = profile.get("is_student")

        if is_student is None:
            return None

        student_terms = [
            "student",
            "students",
            "school",
            "college",
            "scholarship",
        ]

        if not any(term in text for term in student_terms):
            return None

        if is_student:
            return EligibilityCheck(
                "student_status",
                "MATCH",
                "Scheme contains a student/education requirement.",
            )

        return EligibilityCheck(
            "student_status",
            "MISMATCH",
            "Scheme contains a student/education requirement.",
        )

    # ---------------------------------------------------------
    # EDUCATION CLASS
    # ---------------------------------------------------------

    def _check_education_class(self, text, profile):

        education_class = profile.get("education_class")

        if education_class is None:
            return None

        matches = re.findall(
            r"class(?:es)?\s*(\d+)(?:st|nd|rd|th)?\s*(?:to|-)\s*(\d+)",
            text,
        )

        if not matches:
            return None

        for minimum, maximum in matches:
            minimum = int(minimum)
            maximum = int(maximum)

            if minimum <= education_class <= maximum:
                return EligibilityCheck(
                    "education_class",
                    "MATCH",
                    (
                        f"Class {education_class} is within "
                        f"class {minimum}-{maximum}."
                    ),
                )

        return EligibilityCheck(
            "education_class",
            "MISMATCH",
            "User's class does not match the stated class range.",
        )

    # ---------------------------------------------------------
    # TAXPAYER STATUS
    # ---------------------------------------------------------

    def _check_taxpayer(self, text, profile):

        taxpayer = profile.get("is_income_taxpayer")

        if taxpayer is None:
            return None

        taxpayer_requirement = (
            "income taxpayer" in text
            or "income tax payer" in text
            or "income tax payers" in text
        )

        if not taxpayer_requirement:
            return None

        if "should not be" in text or "should not" in text:
            if not taxpayer:
                return EligibilityCheck(
                    "income_taxpayer",
                    "MATCH",
                    "Applicant is not an income taxpayer.",
                )

            return EligibilityCheck(
                "income_taxpayer",
                "MISMATCH",
                "Scheme requires the applicant/guardian not to be an income taxpayer.",
            )

        return None

    # ---------------------------------------------------------
    # LAND OWNERSHIP
    # ---------------------------------------------------------

    def _check_land(self, text, profile):

        land_acres = profile.get("land_acres")

        if land_acres is None:
            return None

        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*acres?",
            text,
        )

        if not matches:
            return None

        if "not own more than" in text:
            limit = float(matches[0])

            if land_acres <= limit:
                return EligibilityCheck(
                    "land_ownership",
                    "MATCH",
                    f"Land ownership {land_acres} acres is within {limit} acres.",
                )

            return EligibilityCheck(
                "land_ownership",
                "MISMATCH",
                f"Land ownership {land_acres} acres exceeds {limit} acres.",
            )

        return None

    # ---------------------------------------------------------
    # REGISTRATION
    # ---------------------------------------------------------

    def _check_registration(self, text, profile):

        registered = profile.get("is_registered")

        if registered is None:
            return None

        registration_terms = [
            "registered",
            "registration",
            "membership",
            "member of",
        ]

        if not any(term in text for term in registration_terms):
            return None

        if registered:
            return EligibilityCheck(
                "registration",
                "MATCH",
                "User reports satisfying the registration/membership requirement.",
            )

        return EligibilityCheck(
            "registration",
            "MISMATCH",
            "Scheme contains a registration/membership requirement.",
        )
    # ---------------------------------------------------------
    # COURSE
    # ---------------------------------------------------------

    def _check_course(self, text, profile):

        course = profile.get("course")

        if not course:
            return None

        course = str(course).lower()

        course_terms = {
            "b.e.": ["b.e.", "b.e ", "bachelor of engineering"],
            "b.tech": ["b.tech", "b.tech.", "bachelor of technology"],
            "engineering": [
                "engineering college",
                "engineering course",
                "engineering degree",
            ],
        }

        terms = course_terms.get(course)

        if not terms:
            return None

        if any(term in text for term in terms):
            return EligibilityCheck(
                "course",
                "MATCH",
                f"Scheme contains a {course} course requirement.",
            )

        return EligibilityCheck(
            "course",
            "MISMATCH",
            f"Scheme does not contain the required {course} course.",
        )

    # ---------------------------------------------------------
    # STUDY YEAR
    # ---------------------------------------------------------

    def _check_study_year(self, text, profile):

        study_year = profile.get("study_year")

        if study_year is None:
            return None

        if not re.search(
            r"\bfirst[- ]year\b|\b1st[- ]year\b",
            text,
        ):
            return None

        if study_year == 1:
            return EligibilityCheck(
                "study_year",
                "MATCH",
                "User is in the first year.",
            )

        return EligibilityCheck(
            "study_year",
            "MISMATCH",
            "Scheme requires a first-year student.",
        )

    # ---------------------------------------------------------
    # CLASS 12 PERCENTAGE
    # ---------------------------------------------------------

    def _check_class_12_percentage(self, text, profile):

        percentage = profile.get("class_12_percentage")

        if percentage is None:
            return None

        patterns = [
            r"at least\s+(\d+(?:\.\d+)?)\s*%",
            r"minimum\s+(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*%\s*marks",
        ]

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:
                required = float(match.group(1))

                if percentage >= required:
                    return EligibilityCheck(
                        "class_12_percentage",
                        "MATCH",
                        (
                            f"Class 12 percentage {percentage}% "
                            f"meets the minimum {required}%."
                        ),
                    )

                return EligibilityCheck(
                    "class_12_percentage",
                    "MISMATCH",
                    (
                        f"Class 12 percentage {percentage}% "
                        f"is below the required {required}%."
                    ),
                )

        return None

    # ---------------------------------------------------------
    # OTHER SCHOLARSHIP
    # ---------------------------------------------------------

    def _check_other_scholarship(self, text, profile):

        receives_other = profile.get(
            "receives_other_scholarship"
        )

        if receives_other is None:
            return None

        scholarship_terms = [
            "other scholarship",
            "other scholarships",
            "any other scholarship",
        ]

        if not any(term in text for term in scholarship_terms):
            return None

        if "must not be receiving" in text:
            if not receives_other:
                return EligibilityCheck(
                    "other_scholarship",
                    "MATCH",
                    "User is not receiving another scholarship.",
                )

            return EligibilityCheck(
                "other_scholarship",
                "MISMATCH",
                "Scheme does not allow another scholarship.",
            )

        return None