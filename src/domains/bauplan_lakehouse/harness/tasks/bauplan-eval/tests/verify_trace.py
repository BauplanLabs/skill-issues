import json
import os
import re
from pathlib import Path

from cli_sdk_map import CLI_TO_SDK, sdk_methods_for

TRACE = Path("/logs/artifacts/trace.jsonl")
DATASET_ENTRY = Path("/logs/artifacts/dataset_entry.json")

_ALL_SDK_METHODS: set[str] = {
    method for methods in CLI_TO_SDK.values() if methods for method in methods
}


def _skills_used() -> set[str]:
    """Return the set of skill names invoked via the `Skill` tool in the trace."""
    skills: set[str] = set()
    for line in TRACE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("kind") != "tool_use" or entry.get("name") != "Skill":
            continue
        skill = (entry.get("input") or {}).get("skill")
        if skill:
            skills.add(skill)
    return skills


def _commands_used() -> set[str]:
    """Return the set of shell commands run via the `Bash` tool in the trace."""
    commands: set[str] = set()
    for line in TRACE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("kind") != "tool_use" or entry.get("name") != "Bash":
            continue
        raw_input = entry.get("input")
        if not isinstance(raw_input, dict):
            continue
        command = raw_input.get("command")
        if command:
            commands.add(command)
    return commands


def _python_sources() -> str:
    """Concatenate the contents of every Python file the agent wrote or edited."""
    parts: list[str] = []
    for line in TRACE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("kind") != "tool_use":
            continue
        name = entry.get("name")
        raw_input = entry.get("input")
        if not isinstance(raw_input, dict):
            continue
        path = raw_input.get("file_path", "")
        if not path.endswith(".py"):
            continue
        if name == "Write":
            parts.append(raw_input.get("content") or "")
        elif name == "Edit":
            parts.append(raw_input.get("new_string") or "")
    return "\n".join(parts)


def _sdk_methods_called(source: str) -> set[str]:
    """Return the set of `<obj>.<method>(` method names called in `source`."""
    return set(re.findall(r"\.(\w+)\s*\(", source))


def verify_expected_skills_used():
    """Assert the required skills were invoked at least once in the trace.

    Normal eval requires every skill in `expected_skills`. During single-skill
    optimization the harness sets `OPTIMIZE_TARGET_SKILL` (via `[verifier.env]`)
    to the skill under optimization; when set, only that skill is gating and the
    other co-listed `expected_skills` are treated as non-gating — a candidate
    cannot make the agent select a skill governed by a different skill's
    description, so a perfect run of the target skill must not be zeroed by a
    sibling skill it cannot influence.
    """
    assert TRACE.exists(), f"trace file not found at {TRACE}"
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    expected = set(json.loads(DATASET_ENTRY.read_text()).get("expected_skills") or [])
    used = _skills_used()

    target = os.environ.get("OPTIMIZE_TARGET_SKILL") or None
    required = {target} if target else expected
    missing = required - used

    print(f"expected skills:  {sorted(expected)}")
    if target:
        print(f"optimize target:  {target} (only this skill is gating)")
        print(f"non-gating:       {sorted(expected - {target})}")
    print(f"skills used:      {sorted(used)}")
    print(f"required:         {sorted(required)}")
    print(f"missing:          {sorted(missing)}")

    assert expected, "dataset entry has no expected_skills to verify against"
    assert not missing, (
        f"required skills not a subset of skills_used; missing={sorted(missing)}"
    )


def verify_expected_commands_used():
    """Assert every `expected_commands` entry was satisfied, in CLI or SDK form.

    `expected_commands` are normalized CLI phrases (e.g. "bauplan branch create").
    An agent may orchestrate via the bauplan CLI, via the Python SDK, or a mix —
    e.g. drive branch/run/merge from the CLI while the pipeline's own models.py
    necessarily does `import bauplan`. So each command is credited if EITHER form
    is present, evaluated PER COMMAND. We deliberately avoid an exclusive
    whole-trace "mode" (the old `if _used_python_sdk(): ... else ...`): writing a
    bauplan model file made every CLI-orchestrated build look like SDK mode and
    then fail for not calling SDK orchestration methods.
    """
    assert TRACE.exists(), f"trace file not found at {TRACE}"
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    expected = list(json.loads(DATASET_ENTRY.read_text()).get("expected_commands") or [])
    assert expected, "dataset entry has no expected_commands to verify against"

    used_commands = _commands_used()
    # Detect SDK calls both in .py files the agent Wrote/Edited AND inline in Bash
    # commands (e.g. `python -c "...client.create_branch(...)"`). Without the
    # latter, an agent that correctly orchestrates via the SDK through Bash gets
    # credited by neither form and loses the command check despite a correct end
    # state — the only observed false-negative across runs.
    sdk_called = _sdk_methods_called(
        _python_sources() + "\n" + "\n".join(used_commands)
    )

    def cli_satisfied(cmd: str) -> bool:
        # `used_commands` holds full shell strings ("bauplan branch create X");
        # the expected phrase is a prefix of one, so match by substring.
        return any(cmd in u for u in used_commands)

    def sdk_satisfied(cmd: str) -> bool:
        methods = sdk_methods_for(cmd)
        return bool(methods) and any(m in sdk_called for m in methods)

    missing = [c for c in expected if not (cli_satisfied(c) or sdk_satisfied(c))]

    print(f"expected commands:  {sorted(set(expected))}")
    print(f"cli commands used:  {sorted(used_commands)}")
    print(f"sdk methods called: {sorted(sdk_called & _ALL_SDK_METHODS)}")
    print(f"missing:            {missing}")

    assert not missing, (
        f"expected_commands satisfied by neither CLI nor SDK form; missing={missing}"
    )


def verify_no_forbidden_actions_used():
    """Assert no `forbidden_actions` regex matches any command in the trace."""
    assert TRACE.exists(), f"trace file not found at {TRACE}"
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    forbidden = json.loads(DATASET_ENTRY.read_text()).get("forbidden_actions") or []
    commands = _commands_used()

    violations: list[tuple[str, str, str]] = []
    for action in forbidden:
        pattern = action.get("pattern")
        reason = action.get("reason", "")
        if not pattern:
            continue
        regex = re.compile(pattern)
        for command in commands:
            if regex.search(command):
                violations.append((pattern, command, reason))

    print(f"forbidden patterns: {[a.get('pattern') for a in forbidden]}")
    print(f"commands used:      {sorted(commands)}")
    print(f"violations:         {violations}")

    assert not violations, f"forbidden_actions matched in trace: {violations}"


if __name__ == "__main__":
    verify_expected_skills_used()
    verify_expected_commands_used()
    verify_no_forbidden_actions_used()
    print("OK")
