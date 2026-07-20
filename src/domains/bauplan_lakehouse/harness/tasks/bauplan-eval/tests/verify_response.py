import asyncio
import json
import os
import re
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

ORACLE = Path("/logs/agent/oracle.txt")
TRACE = Path("/logs/artifacts/trace.jsonl")
DATASET_ENTRY = Path("/logs/artifacts/dataset_entry.json")


_TAG_PREFIXES = ("[assistant]", "[tool_use]", "[tool_result]", "[dataset]")


def _load_entry() -> dict:
    """Load the dataset entry, resolving the literal `<username>` placeholder.

    `<username>` is how the dataset encodes a constraint *about* the user's branch
    generically — it's the only way to write `<username>.main` without knowing the
    runtime user. The verifier resolves it to the live username (e.g. `nsch`)
    before matching. The agent is told to discover its username and run against
    the real branch, so a correct reply names the resolved branch (`nsch.main`)
    and must NOT contain the literal `<username>` token. Matching the resolved
    form is therefore exactly right: it passes the intended reply and rejects one
    that parroted the placeholder back instead of resolving it. Mirrors
    `_load_entry` in verify_bauplan_state. Falls back to raw text if the live
    username can't be read (so the check degrades rather than crashing).
    """
    raw = DATASET_ENTRY.read_text()
    try:
        import bauplan

        username = bauplan.Client().info().user.username
        raw = raw.replace("<username>", username)
    except Exception as exc:
        print(f"[warn] could not resolve <username>, matching raw text: {exc}")
    return json.loads(raw)


def _assistant_text_from_trace() -> str | None:
    """Concatenate every `assistant_text` event from the structured trace.

    This is the source of truth: solve.sh records each assistant text block as a
    discrete `{"kind": "assistant_text", "text": ...}` event, so multi-line
    markdown survives intact. Returns None if the trace is missing/unreadable so
    the caller can fall back to the flattened oracle log.
    """
    if not TRACE.exists():
        return None
    blocks: list[str] = []
    try:
        for line in TRACE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("kind") == "assistant_text":
                blocks.append(event.get("text") or "")
    except (OSError, json.JSONDecodeError):
        return None
    return "\n\n".join(blocks) if blocks else None


def _assistant_text_from_oracle() -> str:
    """Fallback: reconstruct assistant blocks from the flattened oracle log.

    Lossy — the log interleaves `[assistant]`/`[tool_use]`/... lines and a `---`
    result separator, and assistant markdown that contains a `---` divider gets
    truncated. Used only when the structured trace is unavailable.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in ORACLE.read_text().splitlines():
        if line.startswith("[assistant] "):
            if current is not None:
                blocks.append("\n".join(current))
            current = [line[len("[assistant] ") :]]
        elif line.startswith(_TAG_PREFIXES) or line.startswith("---"):
            if current is not None:
                blocks.append("\n".join(current))
                current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks)


def _assistant_text() -> str:
    """Full assistant reply text, from the structured trace when available."""
    from_trace = _assistant_text_from_trace()
    return from_trace if from_trace is not None else _assistant_text_from_oracle()


def verify_response_hard_constraints():
    """Run each `hard_response_constraints` entry over the assistant text."""
    assert ORACLE.exists(), f"oracle output not found at {ORACLE}"
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    end = _load_entry().get("end_conditions") or {}
    constraints = end.get("hard_response_constraints") or []
    if not constraints:
        print("no hard_response_constraints; skipping")
        return 0, 0

    text = _assistant_text()
    print(f"assistant text length: {len(text)}")

    failures: list[str] = []
    for a in constraints:
        kind = a.get("kind")
        value = a.get("value")
        description = a.get("description", "")
        if kind == "contains":
            if not isinstance(value, str) or value not in text:
                failures.append(f"contains({value!r}) failed: {description}")
        elif kind == "matches_regex":
            if not isinstance(value, str) or not re.search(value, text):
                failures.append(f"matches_regex({value!r}) failed: {description}")
        elif kind == "mentions_all":
            if not isinstance(value, list):
                failures.append(
                    f"mentions_all expects a list, got {type(value).__name__}: {description}"
                )
                continue
            missing = [v for v in value if v not in text]
            if missing:
                failures.append(f"mentions_all missing {missing}: {description}")
        else:
            failures.append(f"unsupported constraint kind: {kind}")

    print(f"constraints: {constraints}")
    print(f"failures:    {failures}")

    return len(constraints) - len(failures), len(constraints)


_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge grading a single criterion on an AI "
    "assistant's natural-language reply to a user. You will be given:\n"
    "  1. The user's original request.\n"
    "  2. The assistant's reply to that request.\n"
    "  3. A single rubric criterion describing what a correct reply should "
    "convey.\n\n"
    "Decide whether the assistant's reply meets the rubric criterion. Be "
    "strict but fair: partial credit is not allowed. If the reply clearly "
    "satisfies the criterion, return 1. Otherwise return 0.\n\n"
    "Respond with exactly one character: either '1' or '0', and nothing "
    "else. No explanation, no punctuation, no whitespace commentary."
)


def _build_judge_prompt(
    user_request: str, assistant_reply: str, criterion: str, description: str
) -> str:
    """Format the judge prompt pairing the assistant reply with one rubric criterion."""
    return (
        f"USER REQUEST:\n{user_request}\n\n"
        f"ASSISTANT REPLY:\n{assistant_reply}\n\n"
        f"RUBRIC CRITERION ({criterion}):\n{description}\n\n"
        "Does the assistant reply satisfy this rubric criterion? "
        "Answer with exactly '1' or '0'."
    )


async def _judge_once(prompt: str) -> str:
    """Send the judge prompt to Claude with no tools and return the raw text reply."""
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=os.environ["JUDGE_MODEL"],
        allowed_tools=[],
        system_prompt=_JUDGE_SYSTEM_PROMPT,
    )
    reply_parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    reply_parts.append(block.text)
    return "".join(reply_parts).strip()


def _score(verdict: str) -> int:
    """Parse the judge's verdict into 1 (pass) or 0 (fail)."""
    match = re.search(r"[01]", verdict)
    if match is None:
        return 0
    return 1 if match.group(0) == "1" else 0


def verify_response_soft_constraints():
    """Grade each `soft_response_constraints` rubric entry via an LLM judge; all must score 1."""
    assert ORACLE.exists(), f"oracle output not found at {ORACLE}"
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    entry = _load_entry()
    end = entry.get("end_conditions") or {}
    constraints = end.get("soft_response_constraints") or []
    if not constraints:
        print("no soft_response_constraints; skipping")
        return 0, 0

    user_request = entry.get("prompt") or ""
    assistant_reply = _assistant_text()
    print(f"assistant text length: {len(assistant_reply)}")

    failures: list[str] = []
    for c in constraints:
        criterion = c.get("criterion", "")
        description = c.get("description", "")
        judge_prompt = _build_judge_prompt(
            user_request, assistant_reply, criterion, description
        )
        verdict = asyncio.run(_judge_once(judge_prompt))
        score = _score(verdict)
        print(f"[judge] {criterion}: verdict={verdict!r} score={score}")
        if score != 1:
            failures.append(f"{criterion} failed (verdict={verdict!r}): {description}")

    return len(constraints) - len(failures), len(constraints)


if __name__ == "__main__":
    for fn in (verify_response_hard_constraints, verify_response_soft_constraints):
        print(fn.__name__, fn())
