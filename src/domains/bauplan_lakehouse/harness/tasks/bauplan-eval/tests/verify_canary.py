import hashlib
import json
import re
import subprocess
from pathlib import Path

DATASET_ENTRY = Path("/logs/artifacts/dataset_entry.json")
SKILLS_ROOT = Path("/app/.claude/skills")


def verify_canary_marker():
    """Dump diagnostics about the in-container skill files for the canary.

    Prints each expected skill file's size, sha256, and last lines, any
    gepa-canary marker found, and a recursive grep for the marker. Always
    (0, 0) so it never affects the real reward. Lets the host confirm whether
    the mutated skill reached the container or a cached image was reused
    """
    if not DATASET_ENTRY.exists():
        print("gepa-canary: no dataset entry")
        return 0, 0
    skills = json.loads(DATASET_ENTRY.read_text()).get("expected_skills") or []
    found = {}
    for s in skills:
        p = SKILLS_ROOT / s / "SKILL.md"
        if not p.exists():
            print(f"gepa-canary: MISSING {p}")
            continue
        text = p.read_text()
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        tail = text.rstrip().splitlines()[-2:]
        print(f"gepa-canary: {p} bytes={len(text)} sha256={h} tail={tail!r}")
        m = re.search(r"<!-- gepa-canary: ([0-9a-f]{32}) -->", text)
        if m:
            found[s] = m.group(1)
    grep = subprocess.run(
        ["grep", "-rl", "gepa-canary", str(SKILLS_ROOT)],
        capture_output=True,
        text=True,
    )
    print(f"gepa-canary: grep hits={grep.stdout.split() or 'none'}")
    if found:
        print(f"gepa-canary markers: {found}")
    return 0, 0
