#!/bin/bash
mkdir -p /logs/verifier

VERIFIERS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=/logs/verifier/pytest.log
: > "$LOG"

export IS_SANDBOX=1

# The core reads back exactly two files: /logs/verifier/reward.txt (a single
# float in [0, 1] formatted as `{reward:.4f}`) and /logs/verifier/pytest.log
# (human-readable). reward = passed / total over the check functions below.
# Replace the placeholder check with real ones that grade the agent's result
# against the dataset entry's end_conditions.
PYTHONPATH="${VERIFIERS_DIR}" uv run --project /app python3 - "$LOG" <<'PY'
import json
import sys
import traceback
from pathlib import Path

log_path = Path(sys.argv[1])
reward_path = Path("/logs/verifier/reward.txt")
entry = json.loads(Path("/solution/dataset_entry.json").read_text())


def verify_placeholder() -> tuple[int, int]:
    # <fill in: read /logs/artifacts/trace.jsonl and the backend state, assert
    # the entry's end_conditions hold, and return (passed, total) sub-checks>
    _ = entry
    return 1, 1


# Each check returns (passed, total); a raised exception counts as (0, 1) and a
# total of 0 drops the check from the denominator instead of failing.
checks = [verify_placeholder]

passed = 0
total = 0
lines: list[str] = []
for fn in checks:
    lines.append(f"--- {fn.__name__} ---")
    try:
        p, t = fn()
    except Exception:
        p, t = 0, 1
        lines.append(traceback.format_exc())
    label = "SKIP" if t == 0 else "PASS" if p == t else "FAIL" if p == 0 else "PARTIAL"
    lines.append(f"[{label}] {fn.__name__} ({p}/{t})")
    passed += p
    total += t

reward = (passed / total) if total else 0.0
lines.append(f"summary: {passed}/{total} passed (reward={reward:.4f})")

log_path.write_text("\n".join(lines) + "\n")
reward_path.write_text(f"{reward:.4f}\n")
PY
