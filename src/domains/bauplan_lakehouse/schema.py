import functools
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.core.schema import (
    BenchmarkTask as BaseBenchmarkTask,
)
from src.core.schema import (
    EndConditions,
    StateAssertion,
)
from src.core.schema import (
    validate_task as validate_base_task,
)

# Globals the harness injects into every `validation_script` at runtime (see
# `verify_bauplan_state.verify_bauplan_with_dataset_script`). Stubbed here so
# the standalone type-checker resolves them; the real implementations arrive at
# exec time. Keep the signatures in sync with the harness
_INJECTED_GLOBALS_STUB = (
    "def validate_commit_history(client, user_main, fail, *, namespace, table, "
    "limit=100): ...\n\n"
    "def validate_work_on_novel_branch(client, username, fail, *, namespace, "
    "table): ...\n\n"
)

# Single source of truth for seed data, kept alongside this module:
# `bauplan_datasets[*].tables` lists the permanently-seeded lakehouse tables
# (any other table may exist now but is not guaranteed to persist, so a task
# must never require it to pre-exist); `s3_datasets` lists the public S3 files
# a task may ingest from
_DATASETS_JSON = Path(__file__).parent / "datasets.json"


@functools.lru_cache(maxsize=1)
def seeded_tables() -> frozenset[str]:
    """Fully-qualified 'namespace.table' names guaranteed to exist on `main`.

    Sourced from datasets.json (`bauplan_datasets[*].tables`). Returns an empty
    set if the file is missing so callers can choose to skip the check rather
    than hard-fail in environments without it.
    """
    try:
        data = json.loads(_DATASETS_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(
        t for ds in data.get("bauplan_datasets", []) for t in ds.get("tables", [])
    )


class TaskCategory(str, Enum):
    # READ: no state change
    READ_EXPLORE = "read_explore"
    READ_ASSESSMENT = "read_assessment"
    READ_TEXT_TO_SQL = "read_text_to_sql"
    READ_DATA_MGMT = "read_data_mgmt"
    # WRITE: mutates code, lakehouse state, or both
    WRITE_NEW_PIPELINE = "write_new_pipeline"
    WRITE_ADD_EXPECTATIONS = "write_add_expectations"
    WRITE_MODIFY_PIPELINE = "write_modify_pipeline"
    WRITE_DEBUG_FIX = "write_debug_fix"
    WRITE_DATA_MGMT_OPS = "write_data_mgmt_ops"
    WRITE_INGEST = "write_ingest"
    WRITE_E2E_ANALYSIS = "write_e2e_analysis"


class S3Dataset(BaseModel):
    """A public S3 dataset the harness can pull from to seed test state."""

    name: str
    search_uri: str
    format: Literal["csv", "parquet", "jsonl"]
    files: int
    description: str


@functools.lru_cache(maxsize=1)
def seed_catalog() -> dict[str, S3Dataset]:
    """Seed S3 datasets a task may ingest from, keyed by name.

    Sourced from datasets.json (`s3_datasets`), the same file that lists the
    seeded tables. Raises when the file is missing or malformed: generation
    grounds its prompts on this catalog, so an empty fallback would silently
    degrade every generated task.
    """
    data = json.loads(_DATASETS_JSON.read_text())
    return {
        d.name: d for d in (S3Dataset.model_validate(x) for x in data["s3_datasets"])
    }


class ExpectedTable(BaseModel):
    """Declares a table that must already exist in the lakehouse before the task runs.

    The harness asserts the table is present on `branch` as a precondition;
    nothing in this repo materializes it. Use only for tasks that read an
    existing table (and document the read source); never for ingest tasks.
    """

    table: str = Field(description="Fully-qualified 'namespace.table' name")
    branch: str = "main"


class ExternalDataset(BaseModel):
    """Declares an S3 URI the agent will read from during the task.

    Used for ingest (and ingest-shaped) tasks where the source data lives
    outside the lakehouse. Purely declarative: nothing in the harness
    pre-fetches the file. The agent reads from this URI at solve time.
    """

    s3_uri: str = Field(description="Full s3:// URI the agent will read from")


class AbsentTable(BaseModel):
    """Declares a table that must NOT exist on the agent's `<username>.main`
    at baseline time. The harness drops it (if present) and asserts absence
    before capturing the baseline, so end-condition `table_exists` checks on
    that table cannot be silently satisfied by a leftover from a prior run.

    Branch is implicit: the agent's personal `<username>.main` branch only.
    """

    table: str = Field(description="Fully-qualified 'namespace.table' name")


class InitialState(BaseModel):
    description: str
    branches: list[str] = Field(default_factory=lambda: ["main"])
    expected_tables: list[ExpectedTable] = Field(default_factory=list)
    external_datasets: list[ExternalDataset] = Field(default_factory=list)
    absent_tables: list[AbsentTable] = Field(default_factory=list)


StateAssertionKind = Literal[
    "table_exists",
    "table_absent",
    "row_count",
    "schema_matches",
    "branch_exists",
    "branch_absent",
    "branch_merged",
    "no_change",
]


class BauplanStateAssertion(StateAssertion):
    """State assertion narrowed to the lakehouse vocabulary.

    The Literal `kind` constrains what the generator's LLM tool schema offers
    and what the verifier knows how to check.
    """

    kind: StateAssertionKind
    target: str = Field(description="e.g. 'user.branch:ns.table' or 'main'")


class BauplanEndConditions(EndConditions):
    """End conditions whose state assertions use the lakehouse kinds."""

    state_assertions: list[BauplanStateAssertion] = Field(default_factory=list)


class BauplanBenchmarkTask(BaseBenchmarkTask):
    """Bauplan-specific benchmark task.

    Adds a Bauplan `TaskCategory`, an `InitialState` that knows about branches
    and seeded tables, and lakehouse-narrowed end conditions, on top of the
    generic fields in `src.core.schema.BenchmarkTask`.
    """

    category: TaskCategory
    initial_state: InitialState
    end_conditions: BauplanEndConditions


_WRITE_STATE_KINDS = {
    "table_exists",
    "row_count",
    "schema_matches",
    "branch_exists",
    "branch_merged",
}
_NO_MUTATION_STATE_KINDS = {"no_change", "branch_absent", "table_absent"}

# State-assertion kinds whose `target` MUST be a fully-qualified `branch:ns.table`
# (the verifier parses it via `_split_table_target`). A bare `target: 'main'` makes
# the check parse to an empty table name and pass vacuously, a silent dud
_TABLE_TARGET_KINDS = {"table_exists", "table_absent", "row_count", "schema_matches"}


def validate_task(task: BauplanBenchmarkTask) -> bool:
    """Bauplan-specific semantic checks layered on top of `schema.validate_task`.

    Extra invariants:
    - table-shaped state assertions carry a well-formed `branch:ns.table` target
      (a bare branch like `'main'` would pass vacuously; see `_TABLE_TARGET_KINDS`)
    - READ-category tasks do not assert on state-creating side effects
    - READ-category tasks include at least one write-blocking forbidden_actions
      entry and at least one no-mutation state assertion
    - READ-category tasks keep at least one response constraint (hard OR soft):
      a read task's deliverable IS the reply, so it must be graded. WRITE tasks
      may have neither (they are graded by state_assertions / validation_script).
    """
    if not validate_base_task(task, injected_globals_stub=_INJECTED_GLOBALS_STUB):
        return False

    # Text-level sanity on the free-text fields (the structured checks below only
    # see expected_tables / state_assertions, so a task can carry a seeded
    # expected_tables yet still describe a nonsense scenario). Observed failure
    # mode: the generator leaks its own self-correcting reasoning into
    # `goal`/`prompt` (e.g. "... No — penguins is not seeded. Use an existing
    # table ..."). Depending on a non-seeded table is caught structurally below;
    # legitimately *ingesting* a non-seeded S3 dataset (e.g. penguins) stays
    # valid, so this only rejects the incoherent-text tell.
    prose = f"{task.goal}\n{task.prompt}".lower()
    meta_markers = (
        "wait:",
        "wait,",
        " no —",
        " no -",
        "is not seeded",
        "let me reconsider",
        "on second thought",
        "actually, ",
        "scratch that",
        "hmm,",
    )
    if any(m in prose for m in meta_markers):
        print(
            "REJECT: goal/prompt contains model self-correction / meta-reasoning "
            "text (incoherent task description)",
            file=sys.stderr,
        )
        return False

    # Every table a task requires to pre-exist (initial_state.expected_tables)
    # MUST be a permanently-seeded table. A table that happens to be in the
    # lakehouse now but is not listed in datasets.json is not guaranteed to
    # persist, so building a task on it makes the task flaky. Tasks that need
    # non-seeded data must instead ingest a listed S3 seed file themselves
    # (declare it in external_datasets + absent_tables), never assume it exists.
    seeded = seeded_tables()
    if seeded:  # skip when datasets.json is unavailable rather than hard-fail
        for et in task.initial_state.expected_tables:
            if et.table not in seeded:
                print(
                    f"REJECT: expected_tables requires {et.table!r} which is not a "
                    f"seeded table (not in datasets.json bauplan_datasets[].tables); "
                    f"a task may only depend on seeded tables pre-existing",
                    file=sys.stderr,
                )
                return False

    # Global `main` is never mutated by any task: agents publish only to their
    # personal `<username>.main` (the write workflow merges a feature branch into
    # `<username>.main`, never into the shared `main`). Every entry must declare
    # that invariant as a harness-enforced `no_change(main)` state assertion. A
    # validation_script leak check like `if client.has_table(x, ref='main'): fail`
    # is weaker (it only notices one named table, not any commit to main) and
    # belongs in the state assertion instead.
    if not any(
        sa.kind == "no_change" and sa.target == "main"
        for sa in task.end_conditions.state_assertions
    ):
        print(
            "REJECT: missing state_assertion no_change(main); every task must "
            "assert the shared global `main` is unchanged (agents publish only "
            "to their personal <username>.main)",
            file=sys.stderr,
        )
        return False

    for sa in task.end_conditions.state_assertions:
        if sa.kind in _TABLE_TARGET_KINDS:
            branch, sep, fq = sa.target.partition(":")
            namespace, _, table = fq.partition(".")
            if not (branch and sep and namespace and table):
                print(
                    f"REJECT: {sa.kind} target {sa.target!r} is not a "
                    f"'branch:namespace.table' fqn; it would pass vacuously",
                    file=sys.stderr,
                )
                return False

    if task.category.value.startswith("read_"):
        ec = task.end_conditions
        if not ec.hard_response_constraints and not ec.soft_response_constraints:
            return False
        for sa in task.end_conditions.state_assertions:
            if sa.kind in _WRITE_STATE_KINDS:
                return False
        if not task.forbidden_actions:
            return False
        if not any(
            sa.kind in _NO_MUTATION_STATE_KINDS
            for sa in task.end_conditions.state_assertions
        ):
            return False

    return True
