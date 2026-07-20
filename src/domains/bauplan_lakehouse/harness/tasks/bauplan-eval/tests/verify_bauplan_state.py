import fnmatch
import io
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

DATASET_ENTRY = Path("/logs/artifacts/dataset_entry.json")
BASELINE = Path("/logs/artifacts/bauplan_baseline.json")
PREP_MANIFEST = Path("/logs/artifacts/initial_state_prep.json")

MERGE_PATTERN = re.compile(r"^Merge (?P<source>\S+) into (?P<target>\S+)$")


def validate_commit_history(client, user_main, fail, *, namespace, table, limit=100):
    """Assert that `namespace.table` on `user_main` was *introduced* by merging a
    feature branch — not written directly to `user_main`.

    Injected into every `validation_script`'s globals (see
    `verify_bauplan_with_dataset_script`). Pipeline-build entries call it; entries
    where it doesn't apply (reads, ingests, debug-fixes that mutate in place) just
    don't. Provenance, not mere existence: the destination table must be present
    now on `user_main` AND absent at the pre-merge HEAD of an explicit
    `Merge <feature> into <user_main>` commit. This rules out an agent that wrote
    the table straight onto `user_main` (CTAS / import / insert) and happened to
    also leave an unrelated merge around.

    Relies only on SDK surface confirmed against live bauplan:
      * an explicit feature-branch merge produces a commit whose message matches
        "Merge <source> into <target>" (same MERGE_PATTERN as branch_merged);
      * a commit's first parent (parent_hashes[0]) is the pre-merge HEAD, the
        second (parent_hashes[1]) is the merged-in branch tip;
      * `client.has_table(table, ref=<commit hash>, namespace=...)` queries the
        table's existence at any commit.

    Merges are detected by message, NEVER structurally (parent_hashes == 2):
    every `bauplan run` emits its own 2-parent transactional merge commit
    (message "Run job_id=..."), so a 2-parent test cannot tell a feature->main
    merge from a run-transactional merge.

    Calls `fail()` (the caller's SystemExit helper) on any violation; returns the
    introducing merge commit on success.
    """
    fq = f"{namespace}.{table}"

    # Present-now: the destination table must currently exist on user_main.
    try:
        if not client.has_table(table, ref=user_main, namespace=namespace):
            fail(f"{fq} does not exist on {user_main}")
            return None
    except Exception as exc:
        fail(f"Could not verify {fq} on {user_main}: {exc}")
        return None

    # Pull ONLY merge commits via the server-side message filter (not the full
    # history). filter_by_message is full-match anchored: the pattern must cover
    # the ENTIRE message, so it needs the trailing `.*$` (a bare `^Merge ` prefix
    # matches nothing). Mirrors the working pattern in branch_merged above.
    try:
        commits = list(
            client.get_commits(
                ref=user_main, filter_by_message=r"^Merge.*$", limit=limit
            )
        )
    except Exception as exc:
        fail(f"Could not read commit history on {user_main}: {exc}")
        return None

    feature_merges = []
    for c in commits:
        m = MERGE_PATTERN.match(c.message or "")
        if not m or m["target"] != user_main:
            continue
        src = m["source"]
        # The source must be a real feature branch — never global main, never
        # user_main merging into itself.
        if src in ("main", user_main):
            continue
        feature_merges.append((c, src))

    if not feature_merges:
        fail(
            f"No feature-branch merge commit found on {user_main}: {fq} must be "
            f"built on a branch and merged back, not written directly to {user_main}"
        )
        return None

    # Strict provenance: prove the merge INTRODUCED the table — absent at the
    # merge's first parent (pre-merge HEAD), present now (already confirmed above).
    for commit, src in feature_merges:
        parents = getattr(commit, "parent_hashes", None) or []
        if len(parents) < 2:
            # A "Merge ..." message without two parents isn't a real branch
            # merge; skip it and keep looking for a genuine one.
            continue
        # A bare commit hash is NOT a valid ref for has_table (InvalidRefError);
        # it must be addressed as `<branch>@<hash>` to query that commit's state.
        parent_ref = f"{user_main}@{parents[0]}"
        try:
            existed_before = client.has_table(
                table, ref=parent_ref, namespace=namespace
            )
        except Exception as exc:
            fail(
                f"Could not query pre-merge state of {fq} at {parent_ref}: {exc}"
            )
            return None
        if not existed_before:
            # Absent before this merge, present now => the merge from feature
            # branch `src` introduced it. Workflow confirmed.
            return commit

    fail(
        f"{fq} on {user_main} was not introduced by a feature-branch merge "
        f"(it already existed before every merge commit) — likely written "
        f"directly to {user_main} rather than built on a branch and merged"
    )
    return None


def validate_work_on_novel_branch(client, username, fail, *, namespace, table):
    """Assert the deliverable `namespace.table` was built on a NOVEL agent-created
    branch and left there UNMERGED — the default write-task workflow when the
    prompt gives no merge instruction.

    Injected into every `validation_script`'s globals. DEFAULT write entries call
    this; entries whose prompt EXPLICITLY names a merge target instead use
    `validate_commit_history` against that target.

    Confirmed convention: a write task always works on a fresh branch, never on a
    standing branch. With no merge instruction the agent must NOT merge — the
    result lives on the novel branch and the agent reports its name so a human can
    verify. So this asserts:
      * the deliverable exists on some branch `<username>.<something>` that is NOT
        the personal `<username>.main` (i.e. a fresh, agent-created branch);
      * the deliverable is NOT on `<username>.main` — nothing was merged/written
        there (the `no_change(<username>.main)` state assertion is the canonical
        guard; this is a clearer-message backstop).

    Calls `fail()` on any violation; returns the discovered branch name on success.
    """
    user_main = f"{username}.main"
    fq = f"{namespace}.{table}"
    try:
        names = [b.name for b in client.get_branches()]
    except Exception as exc:
        fail(f"Could not list branches to locate the deliverable: {exc}")
        return None

    # No-merge guard: the result must not have landed on the standing personal main.
    try:
        if client.has_table(table, ref=user_main, namespace=namespace):
            fail(
                f"{fq} exists on {user_main}: a default write task (no merge "
                f"instruction in the prompt) must leave the result on the novel "
                f"branch, unmerged — not publish it to {user_main}"
            )
            return None
    except Exception:
        pass

    novel = [n for n in names if n.startswith(f"{username}.") and n != user_main]
    for bn in novel:
        try:
            if client.has_table(table, ref=bn, namespace=namespace):
                return bn
        except Exception:
            continue

    fail(
        f"{fq} not found on any novel branch created by {username}: a write task "
        f"must build on a fresh branch (off {user_main}) and leave the result there "
        f"for review. User branches seen: {sorted(novel) or '(none)'}"
    )
    return None


def _split_table_target(target: str) -> tuple[str, str, str]:
    """Parse a `branch:namespace.table` target string into its three components."""
    branch, _, fq_table = target.partition(":")
    namespace, _, table = fq_table.partition(".")
    return branch, namespace, table


def _load_entry(client) -> dict:
    """Load the dataset entry, resolving the literal `<username>` placeholder.

    Branch patterns and assertion targets use `<username>.main`; the verifier matches
    them against live branch names like `nsch.main`, so the placeholder must be
    substituted with the real username or every personal-main check silently fails.
    """
    username = client.info().user.username
    return json.loads(DATASET_ENTRY.read_text().replace("<username>", username))


def _row_count_matches(actual: int, expected) -> bool:
    """Compare `actual` against `expected`, where `expected` is an int or a comparison string like '>= 100'."""
    if isinstance(expected, int):
        return actual == expected
    if isinstance(expected, str):
        s = expected.strip()
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if s.startswith(op):
                try:
                    rhs = int(s[len(op) :].strip())
                except ValueError:
                    return False
                return {
                    ">=": actual >= rhs,
                    "<=": actual <= rhs,
                    "==": actual == rhs,
                    "!=": actual != rhs,
                    ">": actual > rhs,
                    "<": actual < rhs,
                }[op]
        try:
            return actual == int(s)
        except ValueError:
            return False
    return False


def verify_bauplan_with_dataset_script():
    """Execute the dataset entry's `validation_script` and pass iff it exits 0."""
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    script = json.loads(DATASET_ENTRY.read_text()).get("validation_script")
    assert script, "dataset entry has no validation_script"

    stderr_buf = io.StringIO()
    exit_code = 0
    try:
        with redirect_stderr(stderr_buf):
            exec(
                compile(script, "<validation_script>", "exec"),
                {
                    "__name__": "__main__",
                    # Opt-in helpers. Default write entries (no merge instruction in
                    # the prompt) call validate_work_on_novel_branch; entries whose
                    # prompt names a merge target call validate_commit_history against
                    # it. Scripts that need neither simply never call them.
                    "validate_commit_history": validate_commit_history,
                    "validate_work_on_novel_branch": validate_work_on_novel_branch,
                },
            )
    except SystemExit as e:
        exit_code = 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)

    if exit_code != 0:
        print(stderr_buf.getvalue(), file=sys.stderr)

    assert exit_code == 0, f"validation_script exited with {exit_code}"


def verify_initial_state():
    """Confirm the branches and seeded tables declared in `initial_state` were present at baseline."""
    import bauplan

    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"
    assert BASELINE.exists(), f"baseline artifact not found at {BASELINE}"

    client = bauplan.Client()
    branches = {b.name for b in client.get_branches()}

    initial = _load_entry(client).get("initial_state") or {}
    baseline = json.loads(BASELINE.read_text())

    baseline_ref = baseline.get("ref")
    baseline_hash = (baseline.get("latest_commit") or {}).get("hash")
    assert baseline_hash, f"baseline artifact has no latest_commit.hash: {baseline}"

    branch_patterns = initial.get("branches") or []
    expected_tables = initial.get("expected_tables") or []
    n_total = len(branch_patterns) + len(expected_tables)
    if n_total == 0:
        print("no initial_state branches or expected_tables; skipping")
        return 0, 0

    failures: list[str] = []

    for branch_pattern in branch_patterns:
        if not any(fnmatch.fnmatchcase(b, branch_pattern) for b in branches):
            failures.append(f"initial branch missing: {branch_pattern}")

    for expected in expected_tables:
        fq_table = expected.get("table", "")
        branch = expected.get("branch", "")
        namespace, _, table = fq_table.partition(".")
        ref = f"{baseline_ref}@{baseline_hash}" if branch == baseline_ref else branch
        tables = {
            t.name for t in client.get_tables(ref=ref, filter_by_namespace=namespace)
        }
        if table not in tables:
            failures.append(f"expected table missing at baseline: {branch}:{fq_table}")

    print(f"initial_state: {initial}")
    print(f"baseline: ref={baseline_ref} hash={baseline_hash}")
    print(f"failures:  {failures}")

    return n_total - len(failures), n_total


def verify_absent_tables_prep():
    """Confirm pre-baseline prep enforced every `absent_tables` entry.

    The agent-runner writes `initial_state_prep.json` after dropping each
    declared table on `<username>.main` and re-verifying absence. This check
    asserts that manifest exists, matches the dataset entry, and that no
    declared absent_tables entry was skipped — guarding against a silent
    failure where prep never ran or targeted the wrong tables.
    """
    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    declared = [
        a.get("table", "")
        for a in (json.loads(DATASET_ENTRY.read_text()).get("initial_state") or {}).get(
            "absent_tables"
        )
        or []
    ]
    if not declared:
        print("no absent_tables declared; skipping prep verification")
        return

    assert PREP_MANIFEST.exists(), (
        f"absent_tables declared but prep manifest missing at {PREP_MANIFEST}"
    )
    manifest = json.loads(PREP_MANIFEST.read_text())
    confirmed = set(manifest.get("confirmed_absent") or [])
    missing = [t for t in declared if t not in confirmed]
    assert not missing, (
        f"absent_tables entries not confirmed absent at baseline: {missing}"
    )
    print(
        f"absent_tables prep verified: declared={declared} "
        f"dropped={manifest.get('dropped')} confirmed_absent={list(confirmed)}"
    )


def verify_end_state_assertions():
    """Run each `state_assertions` entry against the live lakehouse."""
    import bauplan

    assert DATASET_ENTRY.exists(), f"dataset entry not found at {DATASET_ENTRY}"

    client = bauplan.Client()
    branches = {b.name for b in client.get_branches()}

    end = _load_entry(client).get("end_conditions") or {}
    assertions = end.get("state_assertions") or []
    if not assertions:
        print("no state_assertions; skipping")
        return 0, 0

    failures: list[str] = []
    for sa in assertions:
        kind = sa.get("kind")
        target = sa.get("target", "")
        expected = sa.get("expected")
        desc = sa.get("description", "")

        if kind == "branch_exists":
            if not any(fnmatch.fnmatchcase(b, target) for b in branches):
                failures.append(f"branch_exists({target}) failed: {desc}")
        elif kind == "branch_absent":
            matching = [b for b in branches if fnmatch.fnmatchcase(b, target)]
            if matching:
                failures.append(f"branch_absent({target}) matched {matching}: {desc}")
        elif kind == "table_exists":
            branch, ns, tbl = _split_table_target(target)
            tables = {
                t.name for t in client.get_tables(ref=branch, filter_by_namespace=ns)
            }
            if tbl not in tables:
                failures.append(f"table_exists({target}) failed: {desc}")
        elif kind == "table_absent":
            branch, ns, tbl = _split_table_target(target)
            tables = {
                t.name for t in client.get_tables(ref=branch, filter_by_namespace=ns)
            }
            if tbl in tables:
                failures.append(f"table_absent({target}) failed: {desc}")
        elif kind == "row_count":
            branch, ns, tbl = _split_table_target(target)
            result = client.query(f"SELECT COUNT(*) AS n FROM {ns}.{tbl}", ref=branch)
            actual = result.to_pylist()[0]["n"]
            if not _row_count_matches(actual, expected):
                failures.append(
                    f"row_count({target}) expected {expected}, got {actual}: {desc}"
                )
        elif kind == "schema_matches":
            branch, ns, tbl = _split_table_target(target)
            meta = client.get_table(table=f"{ns}.{tbl}", ref=branch)
            actual_schema = {f.name: str(f.type) for f in meta.fields}
            expected_schema = expected or {}
            mismatched = {
                name: (actual_schema.get(name), expected_schema[name])
                for name in expected_schema
                if actual_schema.get(name) != expected_schema[name]
            }
            if mismatched:
                failures.append(
                    f"schema_matches({target}) mismatches={mismatched}: {desc}"
                )
        elif kind == "branch_merged":
            if target not in branches:
                failures.append(
                    f"branch_merged({target}<-{expected}) failed: target branch missing: {desc}"
                )
            elif not isinstance(expected, str) or not expected:
                failures.append(
                    f"branch_merged({target}) failed: expected source branch must be a non-empty string, got {expected!r}: {desc}"
                )
            else:
                matched = False
                for c in client.get_commits(
                    ref=target, filter_by_message=r"^Merge.*$", limit=100
                ):
                    m = MERGE_PATTERN.match(c.message)
                    if m and m["source"] == expected and m["target"] == target:
                        matched = True
                        break
                if not matched:
                    failures.append(
                        f"branch_merged({target}<-{expected}) failed: no matching merge commit on {target}: {desc}"
                    )
        elif kind == "no_change":
            if target not in branches:
                failures.append(f"no_change({target}) failed: branch missing: {desc}")
            elif not BASELINE.exists():
                failures.append(
                    f"no_change({target}) failed: baseline artifact missing at {BASELINE}: {desc}"
                )
            else:
                baseline = json.loads(BASELINE.read_text())
                # Per-ref baselines (main + <username>.main); fall back to the
                # legacy single-ref shape for older baseline artifacts.
                refs = baseline.get("refs") or {
                    baseline.get("ref"): baseline.get("latest_commit")
                }
                if target not in refs:
                    failures.append(
                        f"no_change({target}) failed: no baseline captured for ref "
                        f"{target!r} (have {sorted(refs)}): {desc}"
                    )
                    continue
                username = baseline["username"]
                target_baseline = refs.get(target) or {}
                baseline_date = target_baseline.get("authored_date")
                user_latest = next(
                    iter(
                        client.get_commits(
                            ref=target, filter_by_author_username=username, limit=1
                        )
                    ),
                    None,
                )
                if user_latest is None:
                    print(
                        f"no_change({target}): user={username} has no commits on {target}"
                    )
                elif (
                    baseline_date is not None
                    and str(user_latest.authored_date) > baseline_date
                ):
                    failures.append(
                        f"no_change({target}) failed: user={username} made a commit after baseline "
                        f"(baseline={baseline_date}, user_latest={user_latest.authored_date}, "
                        f"hash={user_latest.ref.hash}): {desc}"
                    )
                else:
                    print(
                        f"no_change({target}): user={username} latest commit {user_latest.ref.hash} authored at {user_latest.authored_date} (<= baseline {baseline_date})"
                    )
        else:
            failures.append(f"unsupported assertion kind: {kind}")

    print(f"assertions: {assertions}")
    print(f"failures:   {failures}")

    return len(assertions) - len(failures), len(assertions)


if __name__ == "__main__":
    failed = False
    for fn in (
        verify_bauplan_with_dataset_script,
        verify_initial_state,
        verify_end_state_assertions,
    ):
        print(f"--- {fn.__name__} ---")
        try:
            result = fn()
        except Exception:
            import traceback

            traceback.print_exc()
            failed = True
        else:
            print(result)
            if isinstance(result, tuple) and result[0] != result[1]:
                failed = True
    if failed:
        sys.exit(1)
    print("OK")
