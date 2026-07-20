from enum import Enum

from src.core.schema import BenchmarkTask
from src.core.schema import validate_task as validate_base_task


class Category(str, Enum):
    # placeholder single value; add the real task categories for this domain
    # here (see src/domains/bauplan_lakehouse/schema.py TaskCategory for a
    # full read/write split)
    EXAMPLE = "example"


class TemplateBenchmarkTask(BenchmarkTask):
    """Domain task model: the generic BenchmarkTask plus a domain category.

    See src/domains/bauplan_lakehouse/schema.py for a full example that also
    narrows StateAssertion.kind to a Literal and adds initial-state models
    (expected/absent tables, external datasets, seeded-resource checks).
    """

    category: Category


def validate_task(task: TemplateBenchmarkTask) -> bool:
    """Domain-specific semantic checks layered on top of core's validate_task."""
    # if this domain's harness injects globals into validation_script at exec
    # time, define a stub of those signatures and thread it in here, e.g.
    # validate_base_task(task, injected_globals_stub=_INJECTED_GLOBALS_STUB)
    return validate_base_task(task)
