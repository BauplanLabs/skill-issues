import csv
import glob
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
    if isinstance(cand, str):
        return cand
    if isinstance(cand, dict):
        cand_map = cast("dict[str, object]", cand)
        if len(cand_map) == 1:
            return str(next(iter(cand_map.values())))
        return str(cand_map.get("current_candidate", ""))
    return str(cand)


def _metrics_from_samples(samples: list[dict]) -> tuple[float | None, float | None]:
    """Mean (cost_usd, time_sec) over a candidate's test-task samples."""
    costs = [
        s.get("agent_report", {}).get("cost_usd")
        for s in samples
        if s.get("agent_report")
    ]
    times = [s.get("timings_sec", {}).get("total") for s in samples]
    costs = [c for c in costs if c is not None]
    times = [t for t in times if t is not None]
    cost = sum(costs) / len(costs) if costs else None
    time = sum(times) / len(times) if times else None
    return cost, time


def _samples_of(test_results: list[dict]) -> list[dict]:
    return [s for r in (test_results or []) for s in (r.get("samples") or [])]


def per_candidate_from_summary(summary_path: Path) -> dict[int, tuple]:
    """{candidate_idx: (cost, time)} from a finished run's optimize-summary JSON."""
    summary = json.loads(summary_path.read_text())
    out: dict[int, tuple] = {}
    for d in summary.get("test_details", []):
        idx = d.get("candidate_idx")
        if idx is None:
            continue
        out[idx] = _metrics_from_samples(_samples_of(d.get("test_results")))
    return out


def per_candidate_from_rundir(run_dir: Path) -> dict[int, tuple]:
    """{candidate_idx: (cost, time)} from a live run's test_progress.jsonl.

    Maps each progress line (keyed by candidate-text hash) to its index via
    gepa_state.bin; falls back to file order if the state isn't readable yet.
    """
    progress = run_dir / "test_progress.jsonl"
    if not progress.exists():
        return {}
    recs = [json.loads(ln) for ln in progress.read_text().splitlines() if ln.strip()]

    idx_by_hash: dict[str, int] = {}
    state_path = run_dir / "gepa_state.bin"
    if state_path.exists():
        try:
            with state_path.open("rb") as f:
                state = pickle.load(f)
            for i, c in enumerate(state.get("program_candidates") or []):
                idx_by_hash[_cand_hash(_state_candidate_text(c))] = i
        except Exception:
            pass

    out: dict[int, tuple] = {}
    for order, rec in enumerate(recs):
        idx = idx_by_hash.get(rec.get("hash"), order)
        out[idx] = _metrics_from_samples(_samples_of(rec.get("test_results")))
    return out


def _resolve_source(src: Path) -> dict[int, tuple]:
    if src.is_dir():
        return per_candidate_from_rundir(src)
    return per_candidate_from_summary(src)


def discover_runs(jobs_glob: str) -> list[tuple[str, Path]]:
    """(label, source) for the newest run under every jobs dir matching the glob.

    Prefers a finished run's optimize-summary; falls back to the newest gepa-* dir.
    """
    found: list[tuple[str, Path]] = []
    for jobs_dir in sorted(glob.glob(jobs_glob)):
        jp = Path(jobs_dir)
        label = jp.parent.parent.name  # <label>/harness/jobs
        summaries = sorted(jp.glob("optimize-summary-*.json"))
        if summaries:
            found.append((label, summaries[-1]))
            continue
        runs = sorted(
            p
            for p in jp.glob("gepa-*")
            if (p / "test_progress.jsonl").exists() or (p / "gepa_state.bin").exists()
        )
        if runs:
            found.append((label, runs[-1]))
    return found


def aggregate(runs: list[tuple[str, Path]]) -> list[dict]:
    """Average cost/time across runs at each candidate index."""
    by_idx: dict[int, dict[str, list]] = {}
    for label, src in runs:
        per = _resolve_source(src)
        for idx, (cost, time) in per.items():
            slot = by_idx.setdefault(idx, {"cost": [], "time": [], "labels": []})
            if cost is not None:
                slot["cost"].append(cost)
            if time is not None:
                slot["time"].append(time)
            slot["labels"].append(label)
    rows = []
    for idx in sorted(by_idx):
        s = by_idx[idx]
        rows.append(
            {
                "candidate_idx": idx,
                "n_runs": len(s["labels"]),
                "mean_cost_usd": (sum(s["cost"]) / len(s["cost"]))
                if s["cost"]
                else None,
                "mean_time_sec": (sum(s["time"]) / len(s["time"]))
                if s["time"]
                else None,
            }
        )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_idx", "n_runs", "mean_cost_usd", "mean_time_sec"])
        for r in rows:
            w.writerow(
                [
                    r["candidate_idx"],
                    r["n_runs"],
                    r["mean_cost_usd"],
                    r["mean_time_sec"],
                ]
            )


def render_png(rows: list[dict], n_runs_total: int, out_png: Path) -> bool:
    """Render mean cost/time per candidate index to a PNG; returns whether it was written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cost_pts = [
        (r["candidate_idx"], r["mean_cost_usd"])
        for r in rows
        if r["mean_cost_usd"] is not None
    ]
    time_pts = [
        (r["candidate_idx"], r["mean_time_sec"])
        for r in rows
        if r["mean_time_sec"] is not None
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    if cost_pts:
        ax1.plot(
            [x for x, _ in cost_pts],
            [y for _, y in cost_pts],
            "o-",
            color="C0",
            label="cost (USD/task)",
        )
    ax1.set_xlabel("GEPA candidate index (iteration; 0 = seed)")
    ax1.set_ylabel("mean cost (USD / task)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    if time_pts:
        ax2.plot(
            [x for x, _ in time_pts],
            [y for _, y in time_pts],
            "s--",
            color="C1",
            label="time (s/task)",
        )
    ax2.set_ylabel("mean wall-clock (s / task)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    xs = [r["candidate_idx"] for r in rows]
    if xs:
        ax1.set_xticks(xs)
    ax1.set_title(
        f"Agent cost & time per GEPA iteration (avg over {n_runs_total} skill runs)"
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    return True


def main(
    sources: list[Path] = typer.Argument(
        None, help="run dirs or optimize-summary JSONs; overrides --jobs-glob"
    ),
    out: Path = typer.Option(Path("jobs/cost_time"), help="output prefix"),
    jobs_glob: str | None = typer.Option(
        None,
        help="glob for jobs dirs to auto-discover the newest run of each, when no sources are given",
    ),
) -> None:
    """Plot mean agent cost and wall-clock time per GEPA candidate index, averaged across runs."""
    if not sources and jobs_glob is None:
        raise SystemExit(
            "pass source run dirs / JSONs, or --jobs-glob to auto-discover"
        )

    if sources:
        runs = [(s.name, s) for s in sources]
    else:
        assert jobs_glob is not None
        runs = discover_runs(jobs_glob)
    runs = [(lbl, src) for lbl, src in runs if src.exists()]
    if not runs:
        raise SystemExit(
            f"no runs found (looked in {jobs_glob}); pass run dirs explicitly"
        )

    print(f"aggregating {len(runs)} run(s): {', '.join(lbl for lbl, _ in runs)}")
    rows = aggregate(runs)
    if not rows:
        raise SystemExit(
            "no per-candidate cost/time samples yet (runs haven't produced a test sweep)"
        )

    prefix = out
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = prefix.with_name(prefix.name + "-curve.csv")
    out_png = prefix.with_name(prefix.name + "-curve.png")
    write_csv(rows, out_csv)
    print(f"CSV written: {out_csv}")
    if render_png(rows, len(runs), out_png):
        print(f"PNG written: {out_png}")


if __name__ == "__main__":
    typer.run(main)
