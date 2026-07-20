import importlib
import os
from enum import Enum
from pathlib import Path
from typing import Any, cast

import typer
import yaml
from dotenv import load_dotenv

from src.core import optimize as opt
from src.core.domain import load_domain_spec
from src.core.schema import PromptSpecificity

app = typer.Typer()

_REPO_ROOT = Path(__file__).parent.parent

# GEPA's reflection LM runs in this process via litellm and reads provider
# credentials from the environment; .env is otherwise only handed to harbor
load_dotenv(_REPO_ROOT / ".env")


class TargetDim(str, Enum):
    """Bucketing dimension for --target coverage goals."""

    CATEGORY = "category"
    SKILL = "skill"


def _domain_root(domain: str) -> Path:
    """Directory of the domain package under src/domains."""
    return Path(__file__).parent / "domains" / domain


def _load_domain_env(domain: str) -> None:
    """Layer the domain's own .env on top of the root one.

    Domains can carry their own secrets.
    """
    load_dotenv(_domain_root(domain) / ".env", override=True)


def _load_config(domain: str) -> Any:
    """Parse the domain's config.yaml (models, paths, upstream skill URLs).

    Every domain package must ship a config.yaml at its root.
    """
    config_path = _domain_root(domain) / "config.yaml"
    if not config_path.exists():
        raise typer.BadParameter(
            f"domain {domain!r} has no config.yaml at {config_path}; every "
            f"domain package must provide one"
        )
    return yaml.safe_load(config_path.read_text())


def _domain_paths(domain: str, config: Any) -> dict[str, Path]:
    """Resolve the config's relative `paths` entries (dataset, harness, skills,
    jobs, dataset_entry) against the domain root."""
    root = _domain_root(domain)
    return {k: root / v for k, v in config["paths"].items()}


@app.command()
def generate(
    domain: str = typer.Argument(..., help="Domain to run"),
    n: int = typer.Option(3, help="Number of tasks to generate"),
    model: str | None = typer.Option(
        None, help="Generator model id (defaults to config models.generator)"
    ),
    overwrite: bool = typer.Option(False, help="Overwrite existing task files"),
    category: list[str] = typer.Option(
        [],
        "--category",
        help="Task category value or domain-defined group prefix; repeatable",
    ),
    specificity: list[PromptSpecificity] = typer.Option(
        [],
        "--specificity",
        help="Allowed prompt specificity level; repeatable",
    ),
    target: list[str] = typer.Option(
        [],
        "--target",
        help="Coverage goal KEY=COUNT (desired total on disk per bucket); "
        "repeatable, supersedes --n and --category",
    ),
    target_dim: TargetDim = typer.Option(
        TargetDim.CATEGORY,
        "--target-dim",
        help="Bucketing dimension for --target keys",
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        help="Maximun number of attemps per generation",
    ),
) -> None:
    """Generate benchmark tasks via an LLM and write them to the dataset."""
    _load_domain_env(domain)
    config = _load_config(domain)
    paths = _domain_paths(domain, config)

    # The domain supplies its concrete generator by convention:
    # src/domains/<domain>/generate.py must define a Generator class
    mod = importlib.import_module(f"src.domains.{domain}.generate")
    generator = cast(Any, mod).Generator()
    # Category specs may be literal values or domain-defined group prefixes;
    # the generator resolves them so the CLI stays domain-agnostic
    categories = generator.parse_category_specs(category) if category else None
    specificities = list(specificity) or None

    # --target switches generation to coverage mode: only the deficit between
    # what is on disk and each KEY=COUNT goal is generated
    targets: dict[str, int] | None = None
    if target:
        if target_dim is TargetDim.SKILL:
            valid_keys = set(generator.fetch_skill_catalog())
        else:
            valid_keys = {c.value for c in generator.all_categories()}
        targets = {}
        for item in target:
            key, sep, count_text = item.partition("=")
            key = key.strip()
            count_text = count_text.strip()
            if not sep or not key or not count_text:
                raise typer.BadParameter(f"--target must be KEY=COUNT, got {item!r}")
            if key not in valid_keys:
                raise typer.BadParameter(
                    f"unknown --target {target_dim.value} {key!r}; "
                    f"valid: {sorted(valid_keys)}"
                )
            try:
                count = int(count_text)
            except ValueError as exc:
                raise typer.BadParameter(
                    f"--target count must be an integer, got {count_text!r}"
                ) from exc
            if count < 0:
                raise typer.BadParameter(f"--target count must be >= 0, got {count}")
            targets[key] = count

    generator.generate_dataset(
        n=n,
        model=model or config["models"]["generator"],
        output_dir=paths["dataset"],
        categories=categories,
        specificities=specificities,
        overwrite=overwrite,
        targets=targets,
        target_dim=target_dim.value,
        max_tokens=config["models"].get("generator_max_tokens"),
        max_attempts=max_attempts,
    )


@app.command()
def optimize(
    domain: str = typer.Argument(..., help="Domain to run"),
    skill: str = typer.Argument(..., help="Skill directory name"),
    num_dataset_entries: int = typer.Option(
        0, help="Cap on the matching-entry pool before splitting; <=0 uses all"
    ),
    n: int = typer.Option(10, help="Max GEPA metric calls for the whole run"),
    parallel: bool = typer.Option(
        False,
        help="Evaluate candidates in parallel; only after the canary confirms the "
        "harness isolates runs",
    ),
    val_frac: float = typer.Option(
        0.15, help="Fraction of the pool held out for GEPA candidate selection"
    ),
    test_frac: float = typer.Option(
        0.15, help="Fraction of the pool sealed as the final test set"
    ),
    split_seed: int = typer.Option(
        0, help="Seed for the deterministic train/val/test shuffle"
    ),
    test_repeats: int = typer.Option(
        1, help="Re-score each test entry N times to average agent variance"
    ),
    resume_dir: Path | None = typer.Option(
        None,
        help="Existing gepa-* run dir to resume from; the caller must reproduce "
        "the original pool and split",
    ),
    primary_only: bool = typer.Option(
        True,
        help="Pool only entries whose primary expected skill (expected_skills[0]) "
        "is this one",
    ),
) -> None:
    """Optimize a skill's SKILL.md via GEPA against the evaluation harness."""
    _load_domain_env(domain)
    config = _load_config(domain)
    paths = _domain_paths(domain, config)
    spec = load_domain_spec(domain)
    os.environ.setdefault("AGENT_MODEL", config["models"]["agent"])
    os.environ.setdefault("JUDGE_MODEL", config["models"]["judge"])
    opt.optimize_skill(
        skill_name=skill,
        num_dataset_entries=num_dataset_entries,
        n=n,
        reflection_model=config["models"]["generator"],
        reflection_objective=spec.reflection_objective,
        reflection_background=spec.reflection_background,
        dataset_dir=paths["dataset"],
        test_harness_dir=paths["harness"],
        skills_path=paths["skills"],
        dataset_entry_path=paths["dataset_entry"],
        jobs_dir=paths["jobs"],
        # harbor receives only the root .env; the domain .env layered above
        # feeds this process, not the container
        env_file=_REPO_ROOT / ".env",
        parallel=parallel,
        val_frac=val_frac,
        test_frac=test_frac,
        split_seed=split_seed,
        test_repeats=test_repeats,
        resume_run_dir=resume_dir,
        primary_only=primary_only,
    )
    print(f"\n=== Diff vs upstream for {skill} ===")
    opt.skill_diff(
        skill_name=skill,
        skills_path=paths["skills"],
        upstream_raw_url=config["upstream_skills_raw"],
    )


@app.command()
def canary(
    domain: str = typer.Argument(..., help="Domain to run"),
    skill: str = typer.Argument(..., help="Skill directory name"),
    entry: str = typer.Argument(
        ...,
        help="Dataset entry filename",
    ),
    n: int = typer.Option(20, help="Max GEPA metric calls"),
) -> None:
    """Prove the eval is real before trusting a run: confirm the mutated skill
    reaches the container and that reward can be driven to 1.0 on one task."""
    _load_domain_env(domain)
    config = _load_config(domain)
    paths = _domain_paths(domain, config)
    spec = load_domain_spec(domain)
    os.environ.setdefault("AGENT_MODEL", config["models"]["agent"])
    os.environ.setdefault("JUDGE_MODEL", config["models"]["judge"])
    opt.run_canary(
        skill_name=skill,
        entry_name=entry,
        n=n,
        dataset_dir=paths["dataset"],
        test_harness_dir=paths["harness"],
        skills_path=paths["skills"],
        dataset_entry_path=paths["dataset_entry"],
        jobs_dir=paths["jobs"],
        env_file=_REPO_ROOT / ".env",
        reflection_model=config["models"]["generator"],
        reflection_objective=spec.reflection_objective,
        reflection_background=spec.reflection_background,
    )


@app.command(name="reset-skills")
def reset_skills(
    domain: str = typer.Argument(..., help="Domain to run"),
) -> None:
    """Restore every local SKILL.md to its upstream version."""
    config = _load_config(domain)
    paths = _domain_paths(domain, config)
    opt.reset_skills(
        skills_path=paths["skills"],
        upstream_raw_url=config["upstream_skills_raw"],
        upstream_api_url=config["upstream_skills_api"],
    )


@app.command(name="skill-diff")
def skill_diff(
    domain: str = typer.Argument(..., help="Domain to run"),
    skill: str = typer.Argument(..., help="Skill directory name"),
) -> None:
    """Print a colorized diff of the local SKILL.md vs upstream."""
    config = _load_config(domain)
    paths = _domain_paths(domain, config)
    opt.skill_diff(
        skill_name=skill,
        skills_path=paths["skills"],
        upstream_raw_url=config["upstream_skills_raw"],
    )


if __name__ == "__main__":
    app()
