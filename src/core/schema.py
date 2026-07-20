import re
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    DIFFICULT = "difficult"


class PromptSpecificity(str, Enum):
    """How precisely the user-facing `prompt` spells out the task.

    Mapped to a user persona so the dataset spans realistic request shapes:
    VAGUE is a novice (short, lay terms, ambiguous, no platform concepts or
    exact identifiers), MEDIUM is a typical user (knows what they want, names
    the object they care about, gives modest detail, leaves some choices
    implicit), DETAILED is an expert (names exact identifiers, types, expected
    output shape, edge cases, and acceptance criteria).
    """

    VAGUE = "vague"
    MEDIUM = "medium"
    DETAILED = "detailed"


class HardResponseConstraint(BaseModel):
    """Deterministic, machine-checkable check on the model's final reply.

    Passes or fails by exact string match / regex / keyword presence: the
    harness evaluates it with no judgment call involved.
    """

    kind: Literal["contains", "matches_regex", "mentions_all"]
    value: str | list[str]
    description: str


class SoftResponseConstraint(BaseModel):
    """Qualitative description of a correct reply, graded by an LLM-as-judge.

    Describes the substance, reasoning, or caveats a good reply should convey
    so an LLM judge can rate whether the model's actual reply meets it:
    captures aspects of correctness that keyword/regex checks cannot.
    """

    criterion: str = Field(
        description=("Short snake_case label for what is being judged")
    )
    description: str = Field(
        description=(
            "Rubric-style description of what a correct reply should convey. "
            "Phrase as substantive guidance for an LLM judge: describe the "
            "content, reasoning, or caveats required, not just keywords. "
            "A judge reads the reply and rates whether it meets this criterion."
        )
    )


class ForbiddenAction(BaseModel):
    pattern: str = Field(
        description="Literal substring or regex to scan tool calls for"
    )
    reason: str


class StateAssertion(BaseModel):
    """Check on backend state after the task completes.

    `kind` is a free-form label at this level; domains subclass with a Literal
    to constrain the vocabulary (the narrowed type also shapes the LLM tool
    schema used during generation).
    """

    kind: str
    target: str = Field(description="Domain-defined locator of the asserted object")
    expected: str | int | dict | None = None
    description: str


class EndConditions(BaseModel):
    hard_response_constraints: list[HardResponseConstraint] = Field(
        default_factory=list
    )
    soft_response_constraints: list[SoftResponseConstraint] = Field(
        default_factory=list
    )
    state_assertions: list[StateAssertion] = Field(default_factory=list)


def validation_script_typechecks(script: str, injected_globals_stub: str = "") -> bool:
    """Write `script` to a temp file and run `uvx ty check` on it.

    Returns True only if ty exits 0 (no syntax or type errors).
    `injected_globals_stub` holds stub definitions for globals the domain's
    harness injects at exec time; prepending them keeps a script that
    legitimately calls those functions from being flagged as referencing an
    undefined name.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(injected_globals_stub + script)
        path = Path(f.name)
    try:
        result = subprocess.run(
            ["uvx", "ty", "check", str(path)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    finally:
        path.unlink(missing_ok=True)


class BenchmarkTask(BaseModel):
    """Generic, domain-agnostic benchmark task.

    Subclasses add domain-specific fields (e.g., a `category` enum and an
    `initial_state` describing seeded state for a particular platform).
    """

    task_name: str
    difficulty: Difficulty = Field(
        description="Relative difficulty of the task for an AI coding assistant"
    )
    prompt_specificity: PromptSpecificity = Field(
        description=(
            "How specific the user-facing `prompt` is, mapped to a "
            "user persona: 'vague' (novice), 'medium' (typical user), "
            "'detailed' (expert). Must match the actual wording of the "
            "`prompt` field. Don't tag a multi-paragraph spec as "
            "'vague' or a one-line ask as 'detailed'."
        )
    )
    goal: str = Field(description="Eval-author's description of success")
    prompt: str = Field(description="Exact user message fed to the agent")
    expected_commands: list[str] = Field(
        default_factory=list,
        description="Reference solution; not used for grading",
    )
    expected_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Names of skills a correct workflow should invoke; "
            "must be a subset of the live skill catalog"
        ),
    )
    end_conditions: EndConditions
    forbidden_actions: list[ForbiddenAction] = Field(default_factory=list)
    validation_script: str = Field(
        description=(
            "Self-contained Python script that, when executed after the task "
            "completes, programmatically verifies the workflow was done "
            "correctly. Must exit with code 0 on success and non-zero on "
            "failure, printing a diagnostic to stderr."
        )
    )
    notes: str | None = None


_TASK_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def validate_task(task: BenchmarkTask, injected_globals_stub: str = "") -> bool:
    """Generic semantic checks that pydantic cannot express.

    Pydantic has already enforced shape and types by the time a `BenchmarkTask`
    instance exists, so this only checks cross-field invariants:

    - `forbidden_actions.pattern` values compile as regex
    - `task_name` is a filesystem-safe snake_case slug
    - every task has at least one `expected_commands` entry (reference solution)
    - `validation_script` parses and typechecks cleanly under `ty`, with
      `injected_globals_stub` prepended for harness-injected globals

    Response constraints are intentionally NOT required here: a check on the
    agent's reply belongs only on tasks whose prompt asks for a reply, so whether
    they are mandatory is a domain/category decision, not a generic one.

    Domain-specific subclasses can layer additional checks on top.
    """
    if not _TASK_NAME_PATTERN.match(task.task_name):
        return False

    for fa in task.forbidden_actions:
        try:
            re.compile(fa.pattern)
        except re.error:
            return False

    if not task.expected_commands:
        return False
    if not task.validation_script.strip():
        return False
    if not validation_script_typechecks(task.validation_script, injected_globals_stub):
        return False

    return True
