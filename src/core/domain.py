import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    """Static domain artifacts consumed by core.

    `reflection_objective` and `reflection_background` feed GEPA's reflection
    prompt; `review_guidance` is appended to the generic semantic-review system
    prompt; `protected_assertions` lists (kind, target) state-assertion pairs
    the reviewer must never drop.
    """

    reflection_objective: str
    reflection_background: str
    review_guidance: str
    protected_assertions: frozenset[tuple[str, str]] = frozenset()


def load_domain_spec(domain: str) -> DomainSpec:
    """Import `src.domains.<domain>.domain` and return its module-level SPEC.

    Follows the same convention-based loading as the generator module. Raises
    TypeError when SPEC is missing or is not a DomainSpec.
    """
    module = importlib.import_module(f"src.domains.{domain}.domain")
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, DomainSpec):
        raise TypeError(
            f"src.domains.{domain}.domain must define a module-level SPEC "
            f"constant of type DomainSpec"
        )
    return spec
