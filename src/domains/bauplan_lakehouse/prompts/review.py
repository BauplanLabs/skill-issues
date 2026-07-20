# Appended to the generic over-constraint review system prompt in
# src/core/semantic_review.py; everything lakehouse-specific the reviewer must
# know lives here.
#
# THIS IS A TEMPLATE. The opening line shows where domain guidance plugs in;
# the actual review rules (which constraints count as over-strict, which
# invariants are protected and must never be dropped, and the per-category hard
# rules) are intentionally not shipped here. Fill in the `FILL HERE` block.
REVIEW_GUIDANCE = (
    "Domain guidance for lakehouse tasks:\n\n"
    "FILL HERE YOUR REVIEW GUIDANCE: the OVER-STRICT cases the reviewer should "
    "flag and drop, the PROTECTED lakehouse-safety invariants it must never "
    "drop no matter what the prompt says, and the HARD RULES about which "
    "constraints each category must keep..."
)
