import difflib
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

from src.core.plot_curve import render_png, write_csv


def _write_dataset_entry(dataset_entry_path: Path, entry: dict[str, Any]) -> None:
    """Write the active task to `dataset_entry_path`, replacing any symlink.

    The harness ships this path as a symlink pointing into a dataset directory
    that does not exist where this process runs, so writing through the link
    would raise. Unlinking first replaces the dangling symlink with a regular
    file holding the entry JSON.
    """
    dataset_entry_path.unlink(missing_ok=True)
    dataset_entry_path.write_text(json.dumps(entry, indent=2) + "\n")


def _assert_canary_reached_container(log_text: str, marker: str, job_name: str) -> None:
    """Raise unless the in-container skill carried this run's canary marker.

    `verify_canary` prints the markers it found under /app/.claude/skills; a
    missing or stale marker means harbor ran a cached image, not the candidate we
    wrote this call
    """
    line = next(
        (ln for ln in log_text.splitlines() if "gepa-canary markers:" in ln), None
    )
    found = re.findall(r"[0-9a-f]{32}", line) if line else []
    if not found:
        raise RuntimeError(
            f"canary {job_name}: no marker in container; the mutated skill did not "
            f"reach /app/.claude/skills (stale/cached image or skills not copied)"
        )
    stale = [f for f in found if f != marker]
    if stale:
        raise RuntimeError(
            f"canary {job_name}: stale marker in container; expected {marker}, got "
            f"{found} (cached image reused instead of rebuilt)"
        )


def _parse_subscores(log_text: str) -> dict[str, float]:
    """Parse guardrail/outcome fractions from the verifier summary line."""
    m = re.search(r"guardrail=(\d+)/(\d+) outcome=(\d+)/(\d+)", log_text)
    if not m:
        return {}
    gp, gt, op, ot = (int(x) for x in m.groups())
    return {
        "guardrail": gp / gt if gt else 0.0,
        "outcome": op / ot if ot else 0.0,
    }


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp, tolerating a trailing Z suffix."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_sec(span: dict[str, Any] | None) -> float | None:
    """Seconds between a span's started_at/finished_at ISO timestamps, or None."""
    if not span:
        return None
    start, end = span.get("started_at"), span.get("finished_at")
    if not start or not end:
        return None
    return (_parse_timestamp(end) - _parse_timestamp(start)).total_seconds()


def _parse_agent_report(trace_path: Path) -> dict[str, Any]:
    """Agent's own end-of-run report from the trace's final `result` record.

    The harness's trace adapter emits cost/duration/turns but drops the SDK
    `usage` object, so raw token counts aren't available here. Returns {} if absent.
    """
    if not trace_path.exists():
        return {}
    final = None
    for line in trace_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "result":
            final = rec
    if not final:
        return {}
    return {
        "cost_usd": final.get("total_cost_usd"),
        "duration_ms": final.get("duration_ms"),
        "num_turns": final.get("num_turns"),
        "is_error": final.get("is_error"),
    }


def _parse_trial_metrics(trial_dir: Path) -> dict[str, Any]:
    """Per-trial wall-clock timings, agent self-report, and any token counts.

    Timings come from each harbor phase in result.json. The agent self-report
    (cost/duration/turns) comes from artifacts/trace.jsonl. `tokens` reads
    result.json's agent_result, which this harness leaves null; the field
    populates automatically for an agent adapter that reports counts.
    """
    result_path = trial_dir / "result.json"
    metrics: dict[str, Any] = {
        "timings_sec": {},
        "tokens": {},
        "agent_report": _parse_agent_report(trial_dir / "artifacts" / "trace.jsonl"),
    }
    if result_path.exists():
        data = json.loads(result_path.read_text())
        ar = data.get("agent_result") or {}
        metrics["timings_sec"] = {
            "total": _duration_sec(
                {
                    "started_at": data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                }
            ),
            "environment_setup": _duration_sec(data.get("environment_setup")),
            "agent_execution": _duration_sec(data.get("agent_execution")),
            "verifier": _duration_sec(data.get("verifier")),
        }
        metrics["tokens"] = {
            "input": ar.get("n_input_tokens"),
            "cache": ar.get("n_cache_tokens"),
            "output": ar.get("n_output_tokens"),
            "cost_usd": ar.get("cost_usd"),
        }
    return metrics


def _make_evaluator(
    skill_path: Path,
    dataset_entry_path: Path,
    test_harness_dir: Path,
    jobs_dir: Path,
    env_file: Path,
) -> Callable[..., tuple[float, dict[str, Any]]]:
    """Return a closure that scores a candidate skill string against one task.

    Writes the candidate to `skill_path` (and, when called with an `example`, that
    task to `dataset_entry_path`), runs the Harbor harness, and returns the reward
    plus side information: the per-check log as `Feedback` and guardrail/outcome
    sub-scores for GEPA's Pareto axes. When `GEPA_CANARY` is set it tags each
    candidate with a unique marker and asserts the marker reached the container.
    """
    canary = bool(os.environ.get("GEPA_CANARY"))
    # The skill under optimization (skill_path is <skills>/<skill_name>/SKILL.md).
    # Exported to the harbor subprocess as OPTIMIZE_TARGET_SKILL so the verifier's
    # expected-skills guardrail grades against only this skill, treating co-listed
    # expected_skills as non-gating: a candidate can't make the agent select a
    # skill governed by a different skill's description
    skill_name = skill_path.parent.name

    def evaluate(
        candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        job_name = f"optimize-run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        if example is not None:
            _write_dataset_entry(dataset_entry_path, example)

        marker = uuid.uuid4().hex if canary else None
        text = candidate.rstrip()
        if marker:
            text += f"\n\n<!-- gepa-canary: {marker} -->"
        skill_path.write_text(text + "\n")

        # The candidate is already written to skill_path, which lives in the
        # Dockerfile build context (`COPY skills/ /app/.claude/skills/`). harbor's
        # layer for that COPY is content-hashed, so changing SKILL.md busts the cache
        # and the rebuilt image carries this candidate; task.toml's skills_dir then
        # copies it to where the agent reads skills. (harbor 0.3.0 has no runtime
        # skill-injection flag; the canary marker check confirms delivery per run)
        subprocess.run(
            [
                "harbor",
                "run",
                "--config",
                "job.yaml",
                "--env-file",
                str(env_file.resolve()),
                "--job-name",
                job_name,
                "--yes",
            ],
            cwd=test_harness_dir,
            env={**os.environ, "OPTIMIZE_TARGET_SKILL": skill_name},
            check=True,
        )

        reward_file = next(
            iter((jobs_dir / job_name).glob("*/verifier/reward.txt")), None
        )
        if reward_file is None:
            raise RuntimeError(
                f"no reward.txt under {jobs_dir / job_name}; the harbor trial likely "
                f"failed before the verifier ran (see {job_name}/*/exception.txt)"
            )
        reward = float(reward_file.read_text().strip())
        # pytest.log holds the per-check pass/fail and the summary line (sub-scores);
        # test-stdout.txt holds each verifier's own diagnostics (and the canary
        # marker echoed by verify_canary). Both are siblings of reward.txt.
        verifier_dir = reward_file.parent
        log_text = (verifier_dir / "pytest.log").read_text()
        stdout_log = verifier_dir / "test-stdout.txt"
        if stdout_log.exists():
            log_text += "\n" + stdout_log.read_text()

        if marker:
            _assert_canary_reached_container(log_text, marker, job_name)

        # The trial dir (parent of the verifier dir) holds result.json and
        # artifacts/trace.jsonl: per-phase timings and the agent's self-report.
        trial_metrics = _parse_trial_metrics(verifier_dir.parent)
        return reward, {
            "Feedback": log_text,
            "scores": _parse_subscores(log_text),
            "trial_metrics": trial_metrics,
        }

    return evaluate


def run_canary(
    skill_name: str,
    entry_name: str,
    n: int,
    dataset_dir: Path,
    test_harness_dir: Path,
    skills_path: Path,
    dataset_entry_path: Path,
    jobs_dir: Path,
    env_file: Path,
    reflection_model: str,
    reflection_objective: str,
    reflection_background: str,
) -> None:
    """Prove the eval is real before trusting any optimization run.

    Runs single-task GEPA on one entry with a unique per-candidate marker and
    `parallel=False`. A stale marker in the container means harbor reused a cached
    image rather than the candidate we wrote. Restores the original SKILL.md so the
    diagnostic never mutates the real skill.
    """
    skill_path = skills_path / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(
            f"SKILL.md for {skill_name!r} not found at {skill_path}"
        )

    entry_path = dataset_dir / entry_name
    if not entry_path.exists():
        raise FileNotFoundError(f"dataset entry not found at {entry_path}")

    entry = json.loads(entry_path.read_text())
    if skill_name not in (entry.get("expected_skills") or []):
        raise ValueError(
            f"{entry_name} does not list {skill_name!r} in expected_skills; "
            f"pick an entry that exercises this skill"
        )

    original_skill = skill_path.read_text()
    _write_dataset_entry(dataset_entry_path, entry)

    rewards: list[float] = []
    os.environ["GEPA_CANARY"] = "1"
    try:
        base_eval = _make_evaluator(
            skill_path, dataset_entry_path, test_harness_dir, jobs_dir, env_file
        )

        def evaluator(
            candidate: str, example: dict[str, Any] | None = None
        ) -> tuple[float, dict[str, Any]]:
            reward, side_info = base_eval(candidate, example)
            rewards.append(reward)
            return reward, side_info

        optimize_anything(
            seed_candidate=original_skill,
            evaluator=evaluator,
            objective=reflection_objective,
            background=reflection_background,
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=n, parallel=False),
                reflection=ReflectionConfig(reflection_lm=reflection_model),
            ),
        )
    finally:
        skill_path.write_text(original_skill)
        os.environ.pop("GEPA_CANARY", None)

    if not rewards:
        print("canary: no evaluations ran")
        return
    print(
        f"\n=== Canary: {len(rewards)} evaluations, markers reached the container "
        f"(no stale-image error raised) ==="
    )
    print(f"reward: min={min(rewards):.4f} max={max(rewards):.4f}")
    if max(rewards) >= 1.0:
        print("reward reached 1.0; the eval can be driven to success, GEPA is real")
    else:
        print(
            f"reward did not reach 1.0 (max={max(rewards):.4f}); inspect "
            f"jobs/*/verifier/pytest.log for the checks still failing"
        )


def _entry_id(entry: dict[str, Any]) -> str:
    """Stable human label for a dataset entry (for split logging/reproducibility)."""
    return str(entry.get("task_name") or entry.get("name") or entry.get("prompt", ""))[
        :60
    ]


def _cand_hash(text: str) -> str:
    """Stable id for a candidate's text, used to match a test score to a candidate.

    Normalized with rstrip() so the string GEPA passes the evaluator at run time
    hashes equal to the one stored in GEPA's state/result later. Must stay in sync
    with the copy in `src/core/plot_curve.py`.
    """
    return hashlib.sha1(text.rstrip().encode("utf-8")).hexdigest()


def _candidate_text(result: Any, i: int) -> str:
    """Plain SKILL.md text for candidate `i` (GEPA stores candidates as dicts)."""
    cand = result.candidates[i]
    if isinstance(cand, str):
        return cand
    key = getattr(result, "_str_candidate_key", None)
    if key and key in cand:
        return cand[key]
    if len(cand) == 1:
        return next(iter(cand.values()))
    return json.dumps(cand)


def _score_on(
    evaluator: Callable[..., tuple[float, dict[str, Any]]],
    candidate: str,
    entries: list[dict[str, Any]],
    test_repeats: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Score one candidate on every entry in `entries`, returning (mean, rows).

    Used for both the held-out test sweep and for re-scoring a rejected candidate
    on the full val set. Each entry is re-run `test_repeats` times (the agent is
    non-deterministic, so a single sample over/under-states generalization). Each
    sample carries reward, sub-scores, wall-clock timings, and the agent's
    self-reported usage.
    """
    rows: list[dict[str, Any]] = []
    for entry in entries:
        samples = []
        for _ in range(max(1, test_repeats)):
            reward, info = evaluator(candidate, entry)
            metrics = info.get("trial_metrics", {})
            samples.append(
                {
                    "reward": reward,
                    "scores": info.get("scores", {}),
                    "timings_sec": metrics.get("timings_sec", {}),
                    "tokens": metrics.get("tokens", {}),
                    "agent_report": metrics.get("agent_report", {}),
                }
            )
        mean = sum(s["reward"] for s in samples) / len(samples)
        rows.append(
            {"entry": _entry_id(entry), "mean_reward": mean, "samples": samples}
        )
    test_mean = sum(r["mean_reward"] for r in rows) / len(rows) if rows else None
    return test_mean, rows


def _split_dataset(
    entries: list[dict[str, Any]], val_frac: float, test_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically shuffle, then split into (train, val, test), all disjoint.

    GEPA reflects on `train` and selects its winning candidate on the held-out
    `val`; `test` is never passed to GEPA and is scored once at the end for an
    unbiased generalization number. The shuffle (fixed `seed`) avoids the
    filename-ordered clustering a contiguous slice would give. Each split is
    guaranteed >=1 entry, so it requires a pool of at least 3.
    """
    n = len(entries)
    if n < 3:
        raise RuntimeError(
            f"3-way split needs >=3 entries, got {n}. Add dataset entries for this "
            f"skill, or run without a held-out val (valset=train) on a smaller pool."
        )
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    # Keep at least one training entry; trim val first, then test, if oversized.
    while n_test + n_val >= n:
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def optimize_skill(
    skill_name: str,
    num_dataset_entries: int,
    n: int,
    reflection_model: str,
    reflection_objective: str,
    reflection_background: str,
    dataset_dir: Path,
    test_harness_dir: Path,
    skills_path: Path,
    dataset_entry_path: Path,
    jobs_dir: Path,
    env_file: Path,
    parallel: bool = False,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    split_seed: int = 0,
    test_repeats: int = 1,
    resume_run_dir: Path | None = None,
    primary_only: bool = True,
) -> None:
    """Optimize a skill via GEPA with a held-out train/val/test split.

    Collects the dataset entries for `skill_name` (by default only those where
    it is the primary expected skill, `expected_skills[0]`; set
    `primary_only=False` to fall back to any entry that merely lists it),
    splits them into disjoint train/val/test (see `_split_dataset`), runs one
    multi-instance GEPA pass (reflect on train, select on the held-out val),
    writes the winning candidate back to SKILL.md, then scores it on the sealed
    test set, the only unbiased generalization number. `reflection_model` is a
    full litellm model id; `reflection_objective` and `reflection_background`
    come from the domain spec. `test_repeats` re-scores each test entry to
    average out agent run-to-run variance. A summary JSON (splits, val score,
    per-entry test rewards) is written under `jobs_dir`.
    """
    skill_path = skills_path / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(
            f"SKILL.md for {skill_name!r} not found at {skill_path}"
        )

    matching: list[dict[str, Any]] = []
    for p in sorted(dataset_dir.glob("*.json")):
        entry = json.loads(p.read_text())
        skills = entry.get("expected_skills") or []
        # Primary = the skill the task is actually built to exercise (first in
        # expected_skills). Co-listed skills are incidental: the validation grades
        # the primary deliverable, so pooling on mere membership optimizes a skill
        # against tasks that don't test it. Default to primary-only.
        if primary_only:
            is_match = bool(skills) and skills[0] == skill_name
        else:
            is_match = skill_name in skills
        if is_match:
            matching.append(entry)
    # num_dataset_entries <= 0 means "use the whole matching pool".
    if num_dataset_entries > 0:
        matching = matching[:num_dataset_entries]

    if not matching:
        scope = "is the primary expected skill of" if primary_only else "list"
        raise RuntimeError(
            f"No dataset entries where {skill_name!r} {scope}; this skill has no "
            f"task that primarily tests it: author dedicated tasks or pass "
            f"--no-primary-only to pool on membership."
        )

    train, val, test = _split_dataset(matching, val_frac, test_frac, split_seed)
    print(
        f"=== Optimizing {skill_name}: {len(train)} train / {len(val)} val / "
        f"{len(test)} test (seed={split_seed}, parallel={parallel}) ==="
    )
    print("  train:", [_entry_id(e) for e in train])
    print("  val:  ", [_entry_id(e) for e in val])
    print("  test: ", [_entry_id(e) for e in test])

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if resume_run_dir is not None:
        # Reuse an existing GEPA run dir: GEPA's initialize_gepa_state() detects
        # gepa_state.bin and resumes from the last saved candidate frontier. The
        # split MUST reproduce the original (same pool + split params) or the
        # loaded valset scores won't align; the caller is responsible for that
        run_dir = Path(resume_run_dir)
        if not (run_dir / "gepa_state.bin").exists():
            raise FileNotFoundError(
                f"--resume-dir {run_dir} has no gepa_state.bin to resume from"
            )
        print(f"=== Resuming GEPA from {run_dir} ===")
    else:
        run_dir = jobs_dir / f"gepa-{skill_name}-{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the split up front so a killed run is still interpretable (which
    # entries were train/val/test) without re-deriving it from these params.
    (run_dir / "split.json").write_text(
        json.dumps(
            {
                "skill": skill_name,
                "split_seed": split_seed,
                "val_frac": val_frac,
                "test_frac": test_frac,
                "test_repeats": test_repeats,
                "train": [_entry_id(e) for e in train],
                "val": [_entry_id(e) for e in val],
                "test": [_entry_id(e) for e in test],
            },
            indent=2,
        )
        + "\n"
    )

    base_evaluator = _make_evaluator(
        skill_path, dataset_entry_path, test_harness_dir, jobs_dir, env_file
    )

    # GEPA only scores candidates on the valset; the held-out TEST score is ours
    # to compute. Capture it the moment GEPA finishes evaluating a candidate on
    # the FULL valset (i.e. once we've seen that candidate against every val
    # entry) so a test point lands next to each val point per iteration;
    # a run killed mid-optimization still has test scores for the candidates it
    # completed. Each result is appended to test_progress.jsonl, keyed by a hash
    # of the candidate text so plot_curve / the final summary can match it back.
    # Test entries are never in `val_ids`, so scoring on test never re-triggers
    # capture. Skipped when parallel=True (the seen-set bookkeeping isn't
    # thread-safe); the post-run loop below scores those candidates instead.
    val_ids = {_entry_id(e) for e in val}
    progress_path = run_dir / "test_progress.jsonl"
    val_seen: dict[str, set[str]] = {}
    captured: dict[str, dict[str, Any]] = {}
    # Every candidate GEPA evaluates passes through here (accepted AND rejected
    # proposals) so this is where we observe the full set of candidates tested
    # (keyed by text hash, kept in first-seen / discovery order: seed is first).
    seen_order: dict[str, str] = {}

    def evaluator(
        candidate: str, example: dict[str, Any] | None = None, **_: Any
    ) -> tuple[float, dict[str, Any]]:
        seen_order.setdefault(_cand_hash(candidate), candidate)
        reward, info = base_evaluator(candidate, example)
        if example is not None and not parallel and _entry_id(example) in val_ids:
            h = _cand_hash(candidate)
            seen = val_seen.setdefault(h, set())
            seen.add(_entry_id(example))
            if seen >= val_ids and h not in captured:
                test_mean, rows = _score_on(
                    base_evaluator, candidate, test, test_repeats
                )
                rec = {"hash": h, "test_reward": test_mean, "test_results": rows}
                captured[h] = rec
                with progress_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"  [iter capture] candidate {len(captured) - 1}: test={test_mean}"
                )
        return reward, info

    result = optimize_anything(
        seed_candidate=skill_path.read_text(),
        evaluator=evaluator,
        dataset=train,
        valset=val,
        objective=reflection_objective,
        background=reflection_background,
        config=GEPAConfig(
            # run_dir persists GEPA's per-iteration run_log.txt + state (the val
            # progression) so the optimization is inspectable and resumable.
            engine=EngineConfig(
                max_metric_calls=n,
                parallel=parallel,
                raise_on_exception=False,
                run_dir=str(run_dir),
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_model),
        ),
    )
    best = result.best_candidate
    best = best if isinstance(best, str) else str(best)
    skill_path.write_text(best if best.endswith("\n") else best + "\n")
    val_scores = list(result.val_aggregate_scores or [])
    print(
        f"\n=== Selected candidate {result.best_idx} of {result.num_candidates} "
        f"(val score {val_scores[result.best_idx] if val_scores else None}) ==="
    )

    # Build the per-candidate val/test curve (x = candidate index, in discovery
    # order, so candidate 0 is always the seed/original skill). We plot every
    # candidate GEPA tested (not just its accepted frontier) so a run whose only
    # proposal was worse than the seed still shows that proposal as a point
    # instead of collapsing to a single dot. GEPA keeps only the frontier in
    # `result`; rejected proposals survive only as the candidate texts our
    # evaluator saw (`seen_order`) and were scored on a minibatch, not the held-out
    # sets. This costs one extra val+test sweep per rejected candidate, beyond GEPA's metric budget.
    frontier_val: dict[str, float | None] = {}
    frontier_parent: dict[str, Any] = {}
    for i in range(result.num_candidates):
        h = _cand_hash(_candidate_text(result, i))
        frontier_val[h] = val_scores[i] if i < len(val_scores) else None
        frontier_parent[h] = result.parents[i] if i < len(result.parents) else None
    best_hash = _cand_hash(_candidate_text(result, result.best_idx))

    # Ordered list of (hash, text) candidates to plot. On a fresh run `seen_order`
    # already holds every candidate in true discovery order (seed first, rejected
    # proposals interleaved exactly where GEPA tried them). But a resumed run won't
    # replay the prior frontier through the evaluator, so first include any
    # frontier candidate missing from this run's `seen_order` (in GEPA index
    # order), then append everything this run actually saw. Guarantees the seed and
    # all accepted candidates always appear.
    ordered: list[tuple[str, str]] = []
    placed: set[str] = set()
    for i in range(result.num_candidates):
        text = _candidate_text(result, i)
        h = _cand_hash(text)
        if h not in seen_order and h not in placed:
            ordered.append((h, text))
            placed.add(h)
    for h, text in seen_order.items():
        if h not in placed:
            ordered.append((h, text))
            placed.add(h)

    n_rejected = sum(1 for h, _ in ordered if h not in frontier_val)
    print(
        f"\n=== Assembling val/test curve: {len(ordered)} candidates tested "
        f"({result.num_candidates} on GEPA frontier, {n_rejected} rejected; "
        f"re-evaluating those on full val+test) ==="
    )
    curve: list[dict[str, Any]] = []
    test_details: list[dict[str, Any]] = []
    best_idx = 0
    for idx, (h, cand_text) in enumerate(ordered):
        # val: reuse GEPA's full-val score for frontier candidates; re-score a
        # rejected proposal (only ever seen on a minibatch) on the full val set.
        if frontier_val.get(h) is not None:
            val_i = frontier_val[h]
        else:
            val_i, _ = _score_on(base_evaluator, cand_text, val, test_repeats)
        # test: reuse the in-run capture if present, else score on the test set.
        rec = captured.get(h)
        if rec is not None:
            test_mean, rows = rec["test_reward"], rec["test_results"]
        else:
            test_mean, rows = _score_on(base_evaluator, cand_text, test, test_repeats)
        # Never drop a point: a failed/empty eval plots as 0.0 so every candidate
        # tested shows up (and the seed always appears, even at reward 0).
        val_i = 0.0 if val_i is None else val_i
        test_mean = 0.0 if test_mean is None else test_mean
        if h == best_hash:
            best_idx = idx
        curve.append(
            {
                "candidate_idx": idx,
                "parent": frontier_parent.get(h),
                "val_reward": val_i,
                "test_reward": test_mean,
                "in_frontier": h in frontier_val,
                "selected": h == best_hash,
            }
        )
        test_details.append({"candidate_idx": idx, "test_results": rows})
        tag = "" if h in frontier_val else " (rejected, re-evaluated)"
        print(f"  candidate {idx}: val={val_i} test={test_mean}{tag}")

    # Roll up timings/agent usage across every test sample of every candidate.
    all_samples = [
        s for d in test_details for r in d["test_results"] for s in r["samples"]
    ]

    def _agg(path: tuple[str, str]) -> dict[str, Any]:
        vals = [
            s[path[0]].get(path[1])
            for s in all_samples
            if s[path[0]].get(path[1]) is not None
        ]
        return {
            "n": len(vals),
            "total": sum(vals) if vals else None,
            "mean": (sum(vals) / len(vals)) if vals else None,
        }

    test_aggregates = {
        "timings_sec": {
            k: _agg(("timings_sec", k))
            for k in ("total", "environment_setup", "agent_execution", "verifier")
        },
        "agent_report": {
            k: _agg(("agent_report", k))
            for k in ("cost_usd", "duration_ms", "num_turns")
        },
        "tokens": {
            k: _agg(("tokens", k)) for k in ("input", "cache", "output", "cost_usd")
        },
    }

    best_point = curve[best_idx] if curve else {}
    summary = {
        "skill": skill_name,
        "reflection_model": reflection_model,
        "n_metric_calls": n,
        "split_seed": split_seed,
        "split": {
            "train": [_entry_id(e) for e in train],
            "val": [_entry_id(e) for e in val],
            "test": [_entry_id(e) for e in test],
        },
        "best_idx": best_idx,
        # num_candidates = total candidates plotted (every candidate GEPA tested);
        # num_frontier_candidates = GEPA's accepted frontier only.
        "num_candidates": len(curve),
        "num_frontier_candidates": result.num_candidates,
        "total_metric_calls": getattr(result, "total_metric_calls", None),
        "best_val_reward": best_point.get("val_reward"),
        "best_test_reward": best_point.get("test_reward"),
        "test_repeats": test_repeats,
        "gepa_run_dir": str(run_dir),
        # `curve` is the plottable series: x=candidate_idx, two y lines
        # (val_reward, test_reward). best_idx marks the deployed skill.
        "curve": curve,
        "test_aggregates": test_aggregates,
        "test_details": test_details,
    }
    summary_path = jobs_dir / f"optimize-summary-{skill_name}-{run_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    rep = test_aggregates["agent_report"]
    print(
        f"\n=== Best candidate {best_idx}: "
        f"val={best_point.get('val_reward')} test={best_point.get('test_reward')} ==="
    )
    print(
        f"=== TEST-sweep usage: cost_usd_total={rep['cost_usd']['total']} "
        f"over {rep['cost_usd']['n']} trials ==="
    )
    print(f"Curve ({len(curve)} points) + summary written to {summary_path}")

    curve_csv = summary_path.with_name(summary_path.stem + "-curve.csv")
    curve_png = summary_path.with_name(summary_path.stem + "-curve.png")
    write_csv(curve, curve_csv)
    if render_png(summary, curve_png):
        print(f"Plot written: {curve_png}")
    print("Best skill wording:\n", best)


def _fetch_upstream_skill(skill_name: str, upstream_raw_url: str) -> str:
    """Return the upstream SKILL.md text for `skill_name`."""
    url = f"{upstream_raw_url}/{skill_name}/SKILL.md"
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def _upstream_skill_names(upstream_api_url: str) -> list[str]:
    """List every skill directory present in the upstream repo."""
    req = urllib.request.Request(
        upstream_api_url, headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req) as resp:
        entries = json.loads(resp.read())
    return sorted(e["name"] for e in entries if e.get("type") == "dir")


def reset_skills(
    skills_path: Path,
    upstream_raw_url: str,
    upstream_api_url: str,
) -> None:
    """Restore every local SKILL.md under `skills_path` to its upstream version.

    Other files in each skill directory are left untouched.
    """
    for name in _upstream_skill_names(upstream_api_url):
        local_path = skills_path / name / "SKILL.md"
        if not local_path.parent.exists():
            print(f"skipping {name}: no local skill directory at {local_path.parent}")
            continue
        local_path.write_text(_fetch_upstream_skill(name, upstream_raw_url))
        print(f"reset {name}")


def skill_diff(
    skill_name: str,
    skills_path: Path,
    upstream_raw_url: str,
) -> None:
    """Print a colorized unified diff of the local SKILL.md versus the upstream one.

    Raises FileNotFoundError when no local SKILL.md exists for `skill_name`.
    """
    local_path = skills_path / skill_name / "SKILL.md"
    if not local_path.exists():
        raise FileNotFoundError(f"no local SKILL.md at {local_path}")

    local_text = local_path.read_text()
    upstream_text = _fetch_upstream_skill(skill_name, upstream_raw_url)

    if local_text == upstream_text:
        print(f"{skill_name}: identical to upstream")
        return

    diff = difflib.unified_diff(
        upstream_text.splitlines(keepends=False),
        local_text.splitlines(keepends=False),
        fromfile=f"upstream/{skill_name}/SKILL.md",
        tofile=f"local/{skill_name}/SKILL.md",
        lineterm="",
    )

    use_color = sys.stdout.isatty()
    GREEN, RED, CYAN, BOLD, RESET = (
        "\033[32m",
        "\033[31m",
        "\033[36m",
        "\033[1m",
        "\033[0m",
    )

    for line in diff:
        if not use_color:
            print(line)
            continue
        if line.startswith("+++") or line.startswith("---"):
            print(f"{BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"{CYAN}{line}{RESET}")
        elif line.startswith("+"):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"{RED}{line}{RESET}")
        else:
            print(line)
