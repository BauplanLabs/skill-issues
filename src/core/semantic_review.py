import json
from typing import Any

from loguru import logger

from src.core.llm import call_tool
from src.core.schema import BenchmarkTask

_REVIEW_SYSTEM_PROMPT = (
    "You audit a single benchmark task for OVER-CONSTRAINT and emit a patch via "
    "the `emit_review` tool. Guiding principle: GRADING MUST MATCH THE PROMPT. "
    "The agent sees ONLY the `prompt` field, never the goal, notes, or "
    "validation_script. A check that grades something the prompt never asked the "
    "agent to do or say is a bug: it fails correct solves.\n\n"
    "Flag and fix these over-constraint patterns:\n"
    "1. REPLY-NOT-REQUESTED: a hard or soft response constraint requires the "
    "agent to say/report/confirm something, but the prompt never asks for any "
    "reply or for that specific content.\n"
    "2. UNGROUNDED NAME: a hard `contains`/`mentions_all` value pins a literal "
    "identifier, name, or number that does not appear verbatim in the prompt, "
    "so the agent cannot know to emit it.\n"
    "3. OVER-STRICT STATE/FORBIDDEN: a state_assertion or forbidden_action is "
    "stricter than the stated task: it forbids or requires something the prompt "
    "does not actually call for. Judge against the PROMPT, not the goal (the "
    "goal often invents requirements the prompt never states).\n\n"
    "Two fix options per issue; choose whichever is MORE APPROPRIATE:\n"
    " (a) DROP the offending constraint (when the check is simply unwarranted), or\n"
    " (b) AMEND THE PROMPT (when the check is desirable and the right fix is to "
    "ask the agent for it); return the full revised prompt in `revised_prompt`. "
    "Keep prompt edits minimal and additive; preserve the task's intent.\n\n"
    "Be conservative: if a constraint is fairly grounded in the prompt, leave it. "
    "Only patch clear over-constraint. If nothing is wrong, return empty drop "
    "lists and null `revised_prompt`."
)


_REVIEW_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_review",
    "description": (
        "Emit the over-constraint patch for this task: which constraints to drop "
        "(by 0-based index into the lists shown) and/or a revised prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "drop_hard_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "0-based indices of hard_response_constraints to remove.",
            },
            "drop_soft_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "0-based indices of soft_response_constraints to remove.",
            },
            "drop_state_assertion_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "0-based indices of state_assertions to remove.",
            },
            "drop_forbidden_action_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "0-based indices of forbidden_actions to remove.",
            },
            "revised_prompt": {
                "type": ["string", "null"],
                "description": (
                    "Full replacement prompt when the fix is to ask the agent "
                    "for the checked behavior. Null to leave the prompt unchanged."
                ),
            },
            "rationale": {
                "type": "string",
                "description": "One or two sentences explaining the patch.",
            },
        },
        "required": [
            "drop_hard_indices",
            "drop_soft_indices",
            "drop_state_assertion_indices",
            "drop_forbidden_action_indices",
            "revised_prompt",
            "rationale",
        ],
    },
}


def _build_user_message(task: BenchmarkTask) -> str:
    """Enumerate the prompt and every grading component with 0-based indices."""
    ec = task.end_conditions
    category = getattr(task, "category", None)
    category_str = getattr(category, "value", category) or "(unknown)"

    def _enumerate(items: list, render) -> str:
        return (
            "\n".join(f"  [{i}] {render(x)}" for i, x in enumerate(items)) or "  (none)"
        )

    return (
        f"CATEGORY: {category_str}\n\n"
        f"GOAL (eval-author's success description, NOT shown to the agent):\n{task.goal}\n\n"
        f"PROMPT (the ONLY text the agent sees):\n{task.prompt}\n\n"
        f"HARD_RESPONSE_CONSTRAINTS:\n"
        + _enumerate(
            ec.hard_response_constraints,
            lambda c: f"kind={c.kind} value={c.value!r} :: {c.description}",
        )
        + "\n\nSOFT_RESPONSE_CONSTRAINTS:\n"
        + _enumerate(
            ec.soft_response_constraints,
            lambda c: f"criterion={c.criterion} :: {c.description}",
        )
        + "\n\nSTATE_ASSERTIONS:\n"
        + _enumerate(
            ec.state_assertions,
            lambda c: f"kind={c.kind} target={c.target!r} :: {c.description}",
        )
        + "\n\nFORBIDDEN_ACTIONS:\n"
        + _enumerate(
            task.forbidden_actions,
            lambda c: f"pattern={c.pattern!r} :: {c.reason}",
        )
        + "\n\nReturn the over-constraint patch via `emit_review`."
    )


def _drop_by_index(items: list, drop: Any, protected: set[int] | None = None) -> list:
    """Return `items` minus the entries at the integer indices in `drop`.

    Invalid / non-integer indices are ignored so a malformed patch can only
    keep items, never crash. Indices in `protected` are never dropped, even if
    the patch lists them.
    """
    drop_set = {i for i in (drop or []) if isinstance(i, int)} - (protected or set())
    return [x for i, x in enumerate(items) if i not in drop_set]


def _protected_state_indices(
    items: list, protected: frozenset[tuple[str, str]]
) -> set[int]:
    """Indices of state assertions whose (kind, target) pair the domain protects.

    The vocabulary lives entirely in the `protected` parameter; this function
    only matches pairs.
    """
    return {
        i
        for i, sa in enumerate(items)
        if (getattr(sa, "kind", None), getattr(sa, "target", None)) in protected
    }


def review_and_fix(
    task: BenchmarkTask,
    model: str,
    guidance: str,
    protected_assertions: frozenset[tuple[str, str]] = frozenset(),
    max_tokens: int | None = None,
) -> tuple[BenchmarkTask, dict | None]:
    """Audit `task` for over-constraint and return `(patched_task, patch)`.

    `guidance` carries the domain's review policy and is appended to the
    generic system prompt; `protected_assertions` lists (kind, target) state
    assertions the patch may never drop. Returns the original task and `None`
    if the model emitted no patch or the patch is a no-op. The patch is applied
    deterministically: constraints listed in the drop indices are removed and
    `prompt` is replaced when a non-empty `revised_prompt` differs from the
    current one. The caller is responsible for re-validating the result.
    """
    system_text = _REVIEW_SYSTEM_PROMPT
    if guidance:
        system_text += "\n\n" + guidance
    raw = call_tool(
        model,
        system_text,
        _build_user_message(task),
        _REVIEW_TOOL_SCHEMA,
        max_tokens=max_tokens,
    )
    if raw is None:
        return task, None
    try:
        patch = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "semantic_review: could not parse emit_review args for {}", task.task_name
        )
        return task, None

    ec = task.end_conditions
    # model_copy keeps the concrete EndConditions subclass a domain may have
    # narrowed the field to, which constructing EndConditions(...) would lose
    new_ec = ec.model_copy(
        update={
            "hard_response_constraints": _drop_by_index(
                ec.hard_response_constraints, patch.get("drop_hard_indices")
            ),
            "soft_response_constraints": _drop_by_index(
                ec.soft_response_constraints, patch.get("drop_soft_indices")
            ),
            "state_assertions": _drop_by_index(
                ec.state_assertions,
                patch.get("drop_state_assertion_indices"),
                protected=_protected_state_indices(
                    ec.state_assertions, protected_assertions
                ),
            ),
        }
    )
    new_forbidden = _drop_by_index(
        task.forbidden_actions, patch.get("drop_forbidden_action_indices")
    )

    update: dict[str, Any] = {
        "end_conditions": new_ec,
        "forbidden_actions": new_forbidden,
    }
    revised = patch.get("revised_prompt")
    if (
        isinstance(revised, str)
        and revised.strip()
        and revised.strip() != task.prompt.strip()
    ):
        update["prompt"] = revised

    patched = task.model_copy(update=update)
    if patched == task:
        return task, None
    return patched, patch
