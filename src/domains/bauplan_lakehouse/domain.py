from src.core.domain import DomainSpec

from .prompts.gepa_reflection import BACKGROUND, OBJECTIVE
from .prompts.review import REVIEW_GUIDANCE

# Standing lakehouse-safety invariants the semantic reviewer must never strip:
# the shared global `main` and the personal `<username>.main` no-change guards
# hold no matter how vague or convention-free the prompt is
_PROTECTED_ASSERTIONS = frozenset(
    {
        ("no_change", "main"),
        ("no_change", "<username>.main"),
    }
)

SPEC = DomainSpec(
    reflection_objective=OBJECTIVE,
    reflection_background=BACKGROUND,
    review_guidance=REVIEW_GUIDANCE,
    protected_assertions=_PROTECTED_ASSERTIONS,
)
