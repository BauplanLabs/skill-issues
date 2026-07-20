#!/bin/bash
set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  exit 1
fi

echo "Running Claude agent on benchmark task..."
export IS_SANDBOX=1
mkdir -p /logs/artifacts
cp /solution/dataset_entry.json /logs/artifacts/dataset_entry.json
cd /app
uv run python - <<'PY'
import asyncio
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import bauplan

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

TRACE = Path("/logs/artifacts/trace.jsonl")
ENTRY = Path("/solution/dataset_entry.json")
BASELINE = Path("/logs/artifacts/bauplan_baseline.json")
PREP_MANIFEST = Path("/logs/artifacts/initial_state_prep.json")


def _table_present(client: bauplan.Client, fq: str, ref: str) -> bool:
    """has_table, but a missing namespace counts as absent rather than an error.

    A fresh `<username>.main` has no output namespace yet, so has_table raises
    NamespaceNotFoundError; for absent_tables that is exactly the desired state
    """
    try:
        return client.has_table(table=fq, ref=ref)
    except bauplan.exceptions.NamespaceNotFoundError:
        return False


def create_user_main() -> None:
    """Create the agent's personal `<username>.main` branch before the run.

    Generated tasks instruct the agent to branch off `<username>.main` and merge
    back into it (see prompts/data_entry.py). Create it from `main` if missing
    so the convention holds on a fresh lakehouse.
    """
    client = bauplan.Client()
    info = client.info()
    assert info.user is not None
    user_main = f"{info.user.username}.main"
    client.create_branch(branch=user_main, from_ref="main", if_not_exists=True)
    print(f"[setup] ensured branch {user_main} exists (from main)")


def prepare_initial_state() -> None:
    """Enforce `initial_state.absent_tables` on both `<username>.main` and global `main`.

    For each declared absent table: drop it on `<username>.main` (idempotent, hard-fail if
    it cannot be made absent), then also drop it on global `main` (best-effort). The
    global-`main` cleanup is required because a buggy skill that merges the output table to
    global `main` leaks it there permanently — the prep used to clean only the personal
    main, so the leak survived and failed every later eval's "absent from global main"
    check regardless of the agent.

    SCOPE: drops only the literal tables listed in `absent_tables` (task outputs that are
    contracted to be absent at start); never inferred, never feature branches.
    """
    entry = json.loads(ENTRY.read_text())
    initial = entry.get("initial_state") or {}
    absent = initial.get("absent_tables") or []

    manifest: dict = {"dropped": [], "confirmed_absent": [], "absent_tables": absent}
    if not absent:
        PREP_MANIFEST.write_text(json.dumps(manifest, indent=2))
        print("[prep] no absent_tables declared; skipping")
        return

    client = bauplan.Client()
    info = client.info()
    assert info.user is not None
    username = info.user.username
    user_main = f"{username}.main"

    # Belt-and-suspenders: never touch main, only ever the personal main.
    assert user_main != "main", "refusing to operate on global main"
    assert user_main.endswith(".main") and "." in user_main, \
        f"refusing to operate on non-personal-main branch: {user_main!r}"

    for entry_at in absent:
        fq = entry_at.get("table") or ""
        namespace, _, table = fq.partition(".")
        assert namespace and table, f"absent_tables entry has malformed FQN: {fq!r}"

        existed = _table_present(client, fq, user_main)
        if existed:
            client.delete_table(
                table=table,
                namespace=namespace,
                branch=user_main,
                if_exists=True,
                commit_body=f"harness prep: drop {fq} for absent_tables baseline",
            )
            manifest["dropped"].append(fq)
            print(f"[prep] dropped {fq} on {user_main} (was present)")
        else:
            print(f"[prep] {fq} already absent on {user_main}")

        # Re-verify absence so a silent drop failure doesn't pass through.
        if _table_present(client, fq, user_main):
            raise RuntimeError(
                f"[prep] FAILED: {fq} still present on {user_main} after drop"
            )
        manifest["confirmed_absent"].append(fq)

    # Also drop the declared outputs on global `main`: a buggy skill that merges to global
    # `main` leaks them there, and the prep used to clean only the personal main, so the leak
    # persisted and failed every later eval's "absent from global main" check. Scoped to the
    # declared absent_tables only; best-effort so a blocked delete on shared main does not
    # abort the whole eval
    for entry_at in absent:
        fq = entry_at.get("table") or ""
        namespace, _, table = fq.partition(".")
        if not _table_present(client, fq, "main"):
            continue
        try:
            client.delete_table(
                table=table,
                namespace=namespace,
                branch="main",
                if_exists=True,
                commit_body=f"harness prep: drop leaked {fq} off global main",
            )
            manifest.setdefault("dropped_on_main", []).append(fq)
            print(f"[prep] dropped leaked {fq} on global main")
        except Exception as exc:
            print(
                f"[prep] WARNING: could not drop {fq} on global main: "
                f"{type(exc).__name__}: {exc}"
            )

    PREP_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[prep] absent_tables enforced: dropped={len(manifest['dropped'])}, "
          f"confirmed_absent={len(manifest['confirmed_absent'])}")


def capture_baseline() -> None:
    client = bauplan.Client()
    username = client.info().user.username
    user_main = f"{username}.main"

    def _commit(ref):
        try:
            latest = next(iter(client.get_commits(ref=ref, limit=1)), None)
        except Exception:
            return "__MISSING__"
        return None if latest is None else {
            "hash": latest.ref.hash,
            "author": str(latest.author),
            "message": latest.message,
            "authored_date": str(latest.authored_date),
        }

    # Per-ref baselines: capture HEAD for every branch a `no_change(<branch>)`
    # end-assertion may target — global `main` AND the agent's personal
    # `<username>.main` (so read tasks can assert their own main is unchanged).
    refs = {}
    for ref in ("main", user_main):
        c = _commit(ref)
        if c != "__MISSING__":
            refs[ref] = c
    baseline = {
        "username": username,
        # Back-compat: top-level ref/latest_commit remain the global-main
        # baseline for older consumers; `refs` is the per-ref map.
        "ref": "main",
        "latest_commit": refs.get("main"),
        "refs": refs,
    }
    BASELINE.write_text(json.dumps(baseline, indent=2))
    print(f"[baseline] user={username} refs={list(refs)} "
          f"main_hash={refs.get('main') and refs['main']['hash']}")


def serialize(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [serialize(x) for x in obj]
    return obj


def write_event(kind: str, payload: dict) -> None:
    with TRACE.open("a") as f:
        f.write(json.dumps({"kind": kind, **payload}, default=str) + "\n")


async def main() -> None:
    entry = json.loads(ENTRY.read_text())
    prompt = entry["prompt"]
    print(f"[dataset] task_name={entry.get('task_name')}")
    print(f"[dataset] prompt={prompt}")

    create_user_main()
    prepare_initial_state()
    capture_baseline()

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=os.environ["AGENT_MODEL"],
        allowed_tools=["Bash", "WebFetch"],
        setting_sources=["project"],
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"[assistant] {block.text}")
                    write_event("assistant_text", {"text": block.text})
                elif isinstance(block, ToolUseBlock):
                    print(f"[tool_use] {block.name} id={block.id} input={block.input}")
                    write_event("tool_use", {
                        "id": block.id,
                        "name": block.name,
                        "input": serialize(block.input),
                    })
                elif isinstance(block, ThinkingBlock):
                    write_event("thinking", {"thinking": block.thinking})
        elif isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, list) else [message.content]
            for block in content:
                if isinstance(block, ToolResultBlock):
                    out = block.content if isinstance(block.content, str) else serialize(block.content)
                    print(f"[tool_result] id={block.tool_use_id} is_error={block.is_error}")
                    write_event("tool_result", {
                        "tool_use_id": block.tool_use_id,
                        "is_error": block.is_error,
                        "content": out,
                    })
        elif isinstance(message, SystemMessage):
            write_event("system", {"subtype": getattr(message, "subtype", None),
                                   "data": serialize(getattr(message, "data", None))})
        elif isinstance(message, ResultMessage):
            print("---")
            print(f"cost_usd={getattr(message, 'total_cost_usd', None)}")
            print(f"duration_ms={getattr(message, 'duration_ms', None)}")
            write_event("result", {
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                "duration_ms": getattr(message, "duration_ms", None),
                "num_turns": getattr(message, "num_turns", None),
                "is_error": getattr(message, "is_error", None),
            })


asyncio.run(main())
PY
