#!/bin/bash
mkdir -p /logs/verifier

VERIFIERS_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG=/logs/verifier/pytest.log
: > "$LOG"

export IS_SANDBOX=1

PYTHONPATH="${VERIFIERS_DIR}" uv run --project /app python3 - "${VERIFIERS_DIR}" "$LOG" <<'PY'
import importlib
import sys
import traceback
from pathlib import Path

verifiers_dir = Path(sys.argv[1])
log_path = Path(sys.argv[2])
reward_path = Path("/logs/verifier/reward.txt")

modules = ["verify_trace", "verify_bauplan_state", "verify_response", "verify_canary"]

# Guardrail checks grade "how" (which skills/commands were used); outcome checks
# grade "whether the task was solved". Tracked separately for GEPA's Pareto axes.
GUARDRAIL_MODULES = {"verify_trace"}

# Each verify_* contributes (passed, total) sub-checks: a returned (p, t) tuple is
# used as-is, a clean return counts as (1, 1), a raised exception as (0, 1). An
# empty/inapplicable section returns (0, 0) so it drops out of the denominator
# instead of failing by construction.
passed = 0
total = 0
g_passed = g_total = o_passed = o_total = 0
lines: list[str] = []

for mod_name in modules:
    lines.append(f"=== {mod_name}.py ===")
    try:
        module = importlib.import_module(mod_name)
    except Exception:
        lines.append(f"[import-error] {mod_name}")
        lines.append(traceback.format_exc())
        lines.append("")
        continue

    fns = [
        (name, getattr(module, name))
        for name in sorted(vars(module))
        if name.startswith("verify_") and callable(getattr(module, name))
    ]
    for name, fn in fns:
        lines.append(f"--- {name} ---")
        try:
            result = fn()
        except Exception:
            p, t = 0, 1
            lines.append(f"[FAIL] {name} ({p}/{t})")
            lines.append(traceback.format_exc())
        else:
            p, t = result if isinstance(result, tuple) else (1, 1)
            if t == 0:
                label = "SKIP"
            elif p == t:
                label = "PASS"
            elif p == 0:
                label = "FAIL"
            else:
                label = "PARTIAL"
            lines.append(f"[{label}] {name} ({p}/{t})")
        passed += p
        total += t
        if mod_name in GUARDRAIL_MODULES:
            g_passed += p
            g_total += t
        else:
            o_passed += p
            o_total += t
    lines.append("")

reward = (passed / total) if total else 0.0
lines.append(
    f"summary: {passed}/{total} passed "
    f"(guardrail={g_passed}/{g_total} outcome={o_passed}/{o_total} reward={reward:.4f})"
)

log_path.write_text("\n".join(lines) + "\n")
reward_path.write_text(f"{reward:.4f}\n")
PY

TEARDOWN="${VERIFIERS_DIR}/teardown/teardown.py"
echo "=== teardown ===" >> "$LOG"
python3 "$TEARDOWN" >> "$LOG" 2>&1 || true
