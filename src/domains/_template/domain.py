from src.core.domain import DomainSpec

from .prompts.gepa_reflection import BACKGROUND, OBJECTIVE
from .prompts.review import REVIEW_GUIDANCE

SPEC = DomainSpec(
    reflection_objective=OBJECTIVE,
    reflection_background=BACKGROUND,
    review_guidance=REVIEW_GUIDANCE,
    # (kind, target) state-assertion pairs the semantic reviewer must never
    # drop; empty until this domain has standing invariants to protect
    protected_assertions=frozenset(),
)
