import json
from pathlib import Path

BASELINE = Path("/logs/artifacts/bauplan_baseline.json")


def teardown() -> None:
    """Delete user-namespaced Bauplan branches whose latest commit post-dates the baseline."""
    import bauplan

    if not BASELINE.exists():
        print(f"no baseline artifact at {BASELINE}; skipping teardown")
        return

    baseline = json.loads(BASELINE.read_text())
    username = baseline.get("username")
    baseline_commit = baseline.get("latest_commit") or {}
    baseline_hash = baseline_commit.get("hash")
    baseline_date = baseline_commit.get("authored_date")

    if not username or not baseline_hash or not baseline_date:
        print(f"baseline missing username/latest_commit fields: {baseline}")
        return

    client = bauplan.Client()

    for branch in client.get_branches():
        # Bauplan convention: user-writable branches are prefixed `<username>.`.
        if not branch.name.startswith(f"{username}."):
            continue

        user_latest = next(
            iter(
                client.get_commits(
                    ref=branch.name, filter_by_author_username=username, limit=1
                )
            ),
            None,
        )
        if user_latest is None:
            continue

        if user_latest.ref.hash == baseline_hash:
            continue
        if str(user_latest.authored_date) <= baseline_date:
            continue

        print(
            f"deleting branch {branch.name} "
            f"(user={username} latest commit {user_latest.ref.hash} @ {user_latest.authored_date} "
            f"> baseline {baseline_hash} @ {baseline_date})"
        )
        try:
            client.delete_branch(branch.name)
        except Exception as e:
            print(f"failed to delete {branch.name}: {e}")


def teardown_user_main() -> None:
    """Delete the agent's personal `<username>.main` branch unconditionally.

    Counterpart to `create_user_main` in solve.sh — that step creates
    `<username>.main` from `main` before the agent runs; this step removes it
    after validation so each run starts from a clean lakehouse, regardless of
    whether the agent merged into it.
    """
    import bauplan

    client = bauplan.Client()
    info = client.info()
    assert info.user is not None
    user_main = f"{info.user.username}.main"
    try:
        client.delete_branch(user_main)
        print(f"deleted branch {user_main}")
    except Exception as e:
        print(f"failed to delete {user_main} (may already be gone): {e}")


if __name__ == "__main__":
    teardown()
    teardown_user_main()
