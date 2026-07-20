import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import cast

import typer


def _cand_hash(text: str) -> str:
    """Hash of a candidate's text. Must match `optimize._cand_hash` exactly."""
    return hashlib.sha1(text.rstrip().encode("utf-8")).hexdigest()


def _state_candidate_text(cand: object) -> str:
    """Plain SKILL.md text for a candidate stored in GEPA state (str or 1-key dict)."""
    if isinstance(cand, str):
        return cand
    if isinstance(cand, dict):
        cand_map = cast("dict[str, object]", cand)
        if len(cand_map) == 1:
            return str(next(iter(cand_map.values())))
        return str(cand_map.get("current_candidate", ""))
    return str(cand)


def _latest_summary(jobs_dir: Path) -> Path | None:
    candidates = sorted(jobs_dir.glob("optimize-summary-*.json"))
    return candidates[-1] if candidates else None


def _latest_gepa_run(jobs_dir: Path) -> Path | None:
    runs = sorted(p for p in jobs_dir.glob("gepa-*") if (p / "gepa_state.bin").exists())
    return runs[-1] if runs else None


def summary_from_gepa_state(run_dir: Path) -> dict:
    """Build a summary dict (same shape as optimize-summary) from a live GEPA run.

    Reads `gepa_state.bin` and derives a val-only curve: one point per discovered
    candidate, val_reward = mean of that candidate's per-val-example subscores.
    test_reward is None (the test sweep only runs after GEPA finishes), so a
    partial/killed run still plots the validation progression it did complete.
    """
    state_path = run_dir / "gepa_state.bin"
    if not state_path.exists():
        raise SystemExit(f"no gepa_state.bin in {run_dir}")
    with state_path.open("rb") as f:
        state = pickle.load(f)

    subscores = state.get("prog_candidate_val_subscores") or []
    parents = state.get("parent_program_for_candidate") or []
    candidates = state.get("program_candidates") or []

    # Test scores captured per candidate during the run (optimize_skill writes
    # one line per committed candidate), keyed by candidate-text hash
    test_by_hash: dict[str, float] = {}
    progress = run_dir / "test_progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                test_by_hash[rec["hash"]] = rec.get("test_reward")

    curve: list[dict] = []
    for i, sub in enumerate(subscores):
        vals = list(sub.values()) if isinstance(sub, dict) else list(sub or [])
        val_mean = sum(vals) / len(vals) if vals else None
        parent = None
        if i < len(parents):
            p = parents[i]
            if isinstance(p, (list, tuple)):
                p = next((x for x in p if x is not None), None)
            parent = p
        test_reward = None
        if i < len(candidates):
            test_reward = test_by_hash.get(
                _cand_hash(_state_candidate_text(candidates[i]))
            )
        curve.append(
            {
                "candidate_idx": i,
                "parent": parent,
                "val_reward": val_mean,
                "test_reward": test_reward,
            }
        )

    best_idx = None
    scored = [
        (c["candidate_idx"], c["val_reward"])
        for c in curve
        if c["val_reward"] is not None
    ]
    if scored:
        best_idx = max(scored, key=lambda t: t[1])[0]

    # Directory name is gepa-<skill>-<YYYYmmdd>-<HHMMSS>; strip prefix + timestamp
    name = run_dir.name
    skill = name[len("gepa-") :].rsplit("-", 2)[0] if name.startswith("gepa-") else name
    return {
        "skill": skill,
        "n_metric_calls": state.get("total_num_evals"),
        "split_seed": None,
        "best_idx": best_idx,
        "num_candidates": len(curve),
        "curve": curve,
        "partial": True,
    }


def write_csv(curve: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_idx", "val_reward", "test_reward", "parent"])
        for row in curve:
            w.writerow(
                [
                    row.get("candidate_idx"),
                    row.get("val_reward"),
                    row.get("test_reward"),
                    row.get("parent"),
                ]
            )


def render_png(summary: dict, out_png: Path) -> bool:
    """Render the val/test reward curve to a PNG; returns whether it was written."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    curve = summary["curve"]

    # Plot each line over only the candidates that actually have that score, so a
    # partial run (test_reward all None) drops the test line instead of erroring
    val_pts = [
        (r["candidate_idx"], r["val_reward"])
        for r in curve
        if r.get("val_reward") is not None
    ]
    test_pts = [
        (r["candidate_idx"], r["test_reward"])
        for r in curve
        if r.get("test_reward") is not None
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    if val_pts:
        ax.plot(
            [x for x, _ in val_pts],
            [y for _, y in val_pts],
            marker="o",
            label="validation reward",
        )
    if test_pts:
        ax.plot(
            [x for x, _ in test_pts],
            [y for _, y in test_pts],
            marker="s",
            label="test reward",
        )
    xs = [r["candidate_idx"] for r in curve]
    ax.set_xlabel("GEPA candidate index (0 = seed skill)")
    ax.set_ylabel("reward")
    ax.set_title(summary.get("skill", "skill"))
    ax.set_ylim(-0.02, 1.02)
    if xs:
        ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return True


def main(
    source: Path | None = typer.Argument(
        None, help="optimize-summary-*.json file, or a gepa-* run directory"
    ),
    out: Path | None = typer.Option(
        None, help="output prefix (default: alongside the source)"
    ),
    jobs_dir: Path | None = typer.Option(
        None,
        help="where to look for the newest summary / gepa run when no source is given",
    ),
) -> None:
    """Plot the val/test reward curve from a GEPA optimization run.

    Two sources, in order of preference: an optimize-summary-*.json (written by
    `optimize_skill` when a run finishes cleanly) has both val and test rewards
    per candidate; a gepa-* run directory's gepa_state.bin (persisted by GEPA
    every iteration) has the per-candidate validation scores only, and is the
    fallback for a run that was killed before the final summary/test sweep, so
    you still get the val curve for whatever candidates completed. Either way
    this writes a dependency-free CSV (candidate_idx, val_reward, test_reward,
    parent) and a PNG of the curve. With no source, picks the newest
    optimize-summary-*.json, else the newest gepa-* run directory, under
    jobs_dir.
    """
    if source is None and jobs_dir is None:
        raise SystemExit("pass a source path or --jobs-dir to search for one")

    src = source
    if src is None:
        assert jobs_dir is not None
        src = _latest_summary(jobs_dir) or _latest_gepa_run(jobs_dir)
    if src is None or not src.exists():
        raise SystemExit(
            f"no summary or gepa run found (looked in {jobs_dir}); pass one explicitly"
        )

    if src.is_dir():
        summary = summary_from_gepa_state(src)
        default_prefix = src / src.name
    else:
        summary = json.loads(src.read_text())
        default_prefix = src.with_suffix("")
    if not summary.get("curve"):
        raise SystemExit(f"{src} has no candidates to plot")

    prefix = out if out is not None else default_prefix
    out_csv = prefix.with_name(prefix.name + "-curve.csv")
    out_png = prefix.with_name(prefix.name + "-curve.png")

    write_csv(summary["curve"], out_csv)
    print(f"CSV written: {out_csv}")
    if render_png(summary, out_png):
        print(f"PNG written: {out_png}")


if __name__ == "__main__":
    typer.run(main)
