# Appended to the generic over-constraint review system prompt in
# src/core/semantic_review.py. Put everything domain-specific the reviewer must
# know here: the protected invariants it must never drop, and the
# category-specific hard rules for which constraints are required vs over-strict.
# See src/domains/bauplan_lakehouse/prompts/review.py for a worked example
REVIEW_GUIDANCE = ""
