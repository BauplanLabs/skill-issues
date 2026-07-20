#!/bin/bash
set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  exit 1
fi

echo "Running oracle agent on benchmark task..."
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


def prepare_initial_state(entry: dict) -> None:
    """Put the backend into the shape `entry["initial_state"]` declares before the agent runs.

    Domain hook: create the resources the task reads, drop leftovers the task
    expects absent, and capture whatever baseline the verifier's no-change
    checks compare against. No-op in this template
    """
    initial = entry.get("initial_state") or {}
    _ = initial  # <fill in: apply the declared preconditions to the live backend>


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

    prepare_initial_state(entry)

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
                    print(f"[tool_use] {block.name} id={block.id}")
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
            write_event("result", {
                "total_cost_usd": getattr(message, "total_cost_usd", None),
                "duration_ms": getattr(message, "duration_ms", None),
                "num_turns": getattr(message, "num_turns", None),
                "is_error": getattr(message, "is_error", None),
            })


asyncio.run(main())
PY
