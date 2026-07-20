import json
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from pydantic import ValidationError

from src.core.domain import DomainSpec
from src.core.llm import call_tool
from src.core.schema import BenchmarkTask, PromptSpecificity
from src.core.semantic_review import review_and_fix


class BenchmarkGenerator(ABC):
    """Abstract base class for benchmark dataset generators.

    Concrete subclasses supply the domain-specific pieces (task schema,
    category list, system prompt, validation logic) while this class owns
    the generation loop and file I/O.
    """

    @property
    @abstractmethod
    def task_class(self) -> type[BenchmarkTask]:
        """The Pydantic model class used to validate and deserialize tasks."""
        ...

    @property
    @abstractmethod
    def spec(self) -> DomainSpec:
        """Static domain artifacts consumed by core: reflection prompt material,
        semantic-review guidance, and the protected state assertions."""
        ...

    @property
    @abstractmethod
    def skills_catalog_url(self) -> str:
        """HTTP(S) URL or local path from which to fetch the skill catalog."""
        ...

    @property
    @abstractmethod
    def prompt(self) -> str:
        """User message sent to the LLM each iteration."""
        ...

    @abstractmethod
    def all_categories(self) -> list:
        """Return the full list of valid category values for this domain."""
        ...

    def parse_category_specs(self, specs: list[str]) -> list:
        """Resolve user-supplied `--category` strings to category values.

        Default behavior: each `spec` must match a category's `.value`
        exactly. Subclasses can override to add group prefixes or aliases.
        Raises `ValueError` on an unknown spec so the CLI fails fast.
        """
        by_name = {getattr(c, "value", c): c for c in self.all_categories()}
        out: list = []
        for spec in specs:
            if spec not in by_name:
                raise ValueError(f"unknown category {spec!r}; valid: {sorted(by_name)}")
            out.append(by_name[spec])
        return out

    @abstractmethod
    def build_system_prompt(
        self,
        existing: list[BenchmarkTask],
        categories: list,
        skill_catalog: list[str],
        specificities: list[PromptSpecificity],
    ) -> str:
        """Assemble the LLM system prompt from domain-specific context."""
        ...

    @abstractmethod
    def validate(self, task: BenchmarkTask) -> bool:
        """Domain-specific semantic validation beyond what Pydantic checks."""
        ...

    @abstractmethod
    def live_check(self, task: BenchmarkTask) -> tuple[bool, str]:
        """Live-environment gate run after validation and semantic review.

        Every domain states its policy explicitly: return `(True, "")` to
        accept, or `(False, reason)` to reject an entry whose verification
        cannot pass against the real backend. A domain with no live backend
        should still implement this and accept unconditionally, making that
        choice visible. `reason` is surfaced when the entry is rejected.
        """
        ...

    def fetch_skill_catalog(self) -> list[str]:
        """Return the current set of skill names.

        Each skill lives in its own directory; the directory name is the skill
        name. When `skills_catalog_url` is an http(s) URL it must point at a
        GitHub repository directory: the listing goes through the GitHub
        contents API, whose JSON response tags each entry with a `type`, so
        only subdirectories are kept. Otherwise the value is treated as a
        local filesystem path and its immediate subdirectories are listed.
        """
        url_or_path = self.skills_catalog_url
        scheme = urlparse(url_or_path).scheme
        if scheme in ("http", "https"):
            req = urllib.request.Request(
                url_or_path, headers={"Accept": "application/vnd.github+json"}
            )
            with urllib.request.urlopen(req) as resp:
                entries = json.loads(resp.read())
            return sorted(e["name"] for e in entries if e.get("type") == "dir")
        return sorted(p.name for p in Path(url_or_path).iterdir() if p.is_dir())

    def _task_key(self, task: object, dim: str) -> object:
        """Bucketing key for a task under the coverage dimension `dim`.

        For "category" the key is the task's category value; for "skill" it is
        the task's primary skill (expected_skills[0]), the same key the
        optimizer's `primary_only` pool filters on.
        """
        if dim == "skill":
            skills = getattr(task, "expected_skills", None) or []
            return skills[0] if skills else None
        cat = getattr(task, "category", None)
        return getattr(cat, "value", cat)

    def _coverage_directive(self, dim: str, key: object, specificity: object) -> str:
        """Per-call instruction pinning this task to one bucket and specificity.

        Used only in coverage-target mode: steers the model to fill the
        currently most-deficient bucket instead of drifting to whatever is
        easiest to invent (which is what skews an unpinned run).
        """
        spec = getattr(specificity, "value", specificity)
        if dim == "skill":
            pin = (
                f'the first entry in expected_skills must be "{key}"; that '
                f"skill is the primary one this task exercises"
            )
            noun = "skill"
        else:
            pin = f'set category="{getattr(key, "value", key)}"'
            noun = "category"
        return (
            f"\n\nCoverage directive (mandatory for this task): {pin}, and set "
            f'prompt_specificity="{spec}". Invent a task that genuinely '
            f"requires that {noun}; do not drift to a different one to make "
            f"the task easier."
        )

    def _resolve_targets(
        self,
        targets: dict[str, int],
        existing: list[BenchmarkTask],
        dim: str,
        valid_keys: dict,
    ) -> dict:
        """Turn `{key: desired_total}` into per-bucket *deficits* for dimension `dim`.

        Existing on-disk entries count toward the goal (top-up semantics), so a
        bucket already at/over its target contributes nothing. `valid_keys` maps
        each acceptable key string to the object stored as the deficit-dict key
        (a category object for "category", the skill string for "skill"). Buckets
        with zero deficit are omitted.
        """
        have = Counter(self._task_key(t, dim) for t in existing)
        remaining: dict = {}
        for key, goal in targets.items():
            if key not in valid_keys:
                raise ValueError(
                    f"unknown target {dim} {key!r}; valid: {sorted(valid_keys)}"
                )
            deficit = max(0, int(goal) - have.get(key, 0))
            if deficit:
                remaining[valid_keys[key]] = deficit
        return remaining

    def generate_dataset(
        self,
        n: int,
        model: str,
        output_dir: Path,
        categories: list | None = None,
        specificities: list[PromptSpecificity] | None = None,
        overwrite: bool = False,
        max_attempts: int | None = None,
        targets: dict[str, int] | None = None,
        target_dim: str = "category",
        max_tokens: int | None = None,
    ) -> list[BenchmarkTask]:
        """Call the LLM until enough tasks pass validation, writing each to output_dir/{task_name}.json.

        Two modes:
        - Freeform (default): generate `n` tasks, each freely picking any
          category in `categories` / specificity in `specificities`.
        - Coverage-target (`targets={key: desired_total}`): generate exactly the
          deficit needed to bring each bucket up to its target (existing entries
          count), pinning every call to the most-deficient bucket and cycling
          specificity so buckets fill evenly. `n` is ignored; the total is
          derived from the deficits. `target_dim` selects the bucketing key:
          "category" (task category) or "skill" (primary skill, expected_skills[0]).
        """
        logger.debug(f"Generating {n} dataset entries with model '{model}'.")
        logger.debug(f"Entries will be saved in '{output_dir}'.")

        output_dir.mkdir(parents=True, exist_ok=True)
        allowed_specs = specificities or list(PromptSpecificity)

        existing = self._load_existing_tasks(output_dir)
        taken_names = {t.task_name for t in existing}
        skill_catalog = self.fetch_skill_catalog()
        valid_skills = set(skill_catalog)
        logger.debug(
            "{} existing tasks on disk, {} skills in catalog",
            len(existing),
            len(skill_catalog),
        )

        remaining: dict | None = None
        if targets:
            if target_dim not in ("category", "skill"):
                raise ValueError(
                    f"target_dim must be 'category' or 'skill', got {target_dim!r}"
                )
            by_value = {getattr(c, "value", c): c for c in self.all_categories()}
            # Deficit dict is keyed by the same value _task_key returns, so the
            # per-emission credit check below lines up: category-value strings
            # for "category", skill-name strings for "skill".
            valid_keys = (
                {s: s for s in skill_catalog}
                if target_dim == "skill"
                else {v: v for v in by_value}
            )
            remaining = self._resolve_targets(targets, existing, target_dim, valid_keys)
            logger.info(
                "coverage mode ({}): deficits {}",
                target_dim,
                {getattr(k, "value", k): v for k, v in remaining.items()},
            )
            # "category" mode restricts the system prompt to the targeted
            # categories; "skill" mode leaves category free (skill is implied).
            if target_dim == "category":
                allowed_cats = [by_value[v] for v in remaining] or self.all_categories()
            else:
                allowed_cats = self.all_categories()
            n = sum(remaining.values())
            if n == 0:
                logger.info("coverage targets already met; nothing to generate")
                return []
        else:
            allowed_cats = categories or self.all_categories()

        cap = max_attempts if max_attempts is not None else n * (4 if targets else 3)
        logger.debug(f"Max attempts allowed: {cap}.")

        system_text = self.build_system_prompt(
            existing, allowed_cats, skill_catalog, allowed_specs
        )
        tool_schema = {
            "name": "emit_task",
            "description": "Emit exactly one new BenchmarkTask as structured JSON.",
            "parameters": self.task_class.model_json_schema(),
        }

        generated: list[BenchmarkTask] = []
        attempts = 0
        spec_cycle = 0

        while len(generated) < n:
            if attempts >= cap:
                raise RuntimeError(
                    f"Exceeded max_attempts={cap} with only "
                    f"{len(generated)}/{n} tasks validated"
                )
            attempts += 1
            logger.debug(f"Attempting generation #{attempts}")

            directive = ""
            if remaining is not None:
                # Steer toward whichever bucket is furthest from its target.
                pinned = max(remaining, key=lambda k: remaining[k])
                pinned_spec = allowed_specs[spec_cycle % len(allowed_specs)]
                spec_cycle += 1
                directive = self._coverage_directive(target_dim, pinned, pinned_spec)

            user_msg = (
                f"{self.prompt}\n\n"
                f"Already-used task names (avoid duplicates): "
                f"{sorted(taken_names) or '(none)'}"
                f"{directive}"
            )

            raw_args = call_tool(
                model, system_text, user_msg, tool_schema, max_tokens=max_tokens
            )
            if raw_args is None:
                logger.info("attempt {}/{}: model emitted no tool call", attempts, cap)
                continue

            try:
                payload = json.loads(raw_args)
                task = self.task_class.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.info(
                    "attempt {}/{}: payload rejected ({})",
                    attempts,
                    cap,
                    type(exc).__name__,
                )
                continue

            task_cat = getattr(task, "category", None)
            task_key = None
            if remaining is not None:
                # Credit whatever needed bucket the model actually emitted (the
                # directive steers, but an on-plan drift still counts); reject
                # anything already at quota or off-plan.
                task_key = self._task_key(task, target_dim)
                if remaining.get(task_key, 0) <= 0:
                    logger.info(
                        "rejected {}: bucket {!r} has no remaining deficit",
                        task.task_name,
                        task_key,
                    )
                    continue
            elif task_cat not in allowed_cats:
                logger.info(
                    "rejected {}: category {!r} not allowed",
                    task.task_name,
                    getattr(task_cat, "value", task_cat),
                )
                continue
            if getattr(task, "prompt_specificity", None) not in allowed_specs:
                logger.info(
                    "rejected {}: specificity {!r} not allowed",
                    task.task_name,
                    getattr(task.prompt_specificity, "value", None),
                )
                continue
            if task.task_name in taken_names:
                logger.info("rejected {}: task name already used", task.task_name)
                continue
            unknown_skills = set(task.expected_skills) - valid_skills
            if unknown_skills:
                logger.info(
                    "rejected {}: expected_skills {} not in the catalog",
                    task.task_name,
                    sorted(unknown_skills),
                )
                continue
            if not self.validate(task):
                logger.info("rejected {}: failed domain validation", task.task_name)
                continue

            task = self._semantic_review_and_fix(task, model, max_tokens)

            ok, reason = self.live_check(task)
            if not ok:
                logger.warning("live_check rejected {}: {}", task.task_name, reason)
                continue

            path = output_dir / f"{task.task_name}.json"
            if path.exists() and not overwrite:
                logger.info(
                    "skipped {}: {} exists and overwrite is off",
                    task.task_name,
                    path.name,
                )
                continue

            path.write_text(json.dumps(task.model_dump(mode="json"), indent=2) + "\n")
            taken_names.add(task.task_name)
            generated.append(task)
            logger.info(
                "accepted {} ({}/{} generated, attempt {}/{})",
                task.task_name,
                len(generated),
                n,
                attempts,
                cap,
            )
            if remaining is not None:
                remaining[task_key] -= 1
                if remaining[task_key] <= 0:
                    del remaining[task_key]

        logger.info(
            "done: {}/{} tasks generated in {} attempts", len(generated), n, attempts
        )
        return generated

    def _semantic_review_and_fix(
        self, task: BenchmarkTask, model: str, max_tokens: int | None = None
    ) -> BenchmarkTask:
        """Run the LLM over-constraint review, applying its patch only if it re-validates.

        The reviewer drops grading checks the `prompt` never warranted and/or
        amends the prompt, steered by the domain's `spec.review_guidance` and
        `spec.protected_assertions`. The result is re-run through
        `self.validate`, so a patch that would break a domain invariant is
        discarded and the original, already-valid task is kept. Any failure in
        the review itself is non-fatal: generation proceeds with the original.
        """
        try:
            patched, patch = review_and_fix(
                task,
                model,
                guidance=self.spec.review_guidance,
                protected_assertions=self.spec.protected_assertions,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.exception(
                "semantic review failed for {}; keeping original", task.task_name
            )
            return task

        if patch is None or patched == task:
            return task
        if not self.validate(patched):
            logger.warning(
                "semantic review patch for {} failed re-validation; discarding: {}",
                task.task_name,
                patch.get("rationale", ""),
            )
            return task
        logger.info(
            "semantic review revised {}: {}", task.task_name, patch.get("rationale", "")
        )
        return patched

    def _load_existing_tasks(self, output_dir: Path) -> list[BenchmarkTask]:
        """Read every *.json task already in `output_dir` and return the valid ones."""
        if not output_dir.exists():
            logger.warning("Task output directory does not exist: {}", output_dir)
            return []
        tasks: list[BenchmarkTask] = []
        for path in sorted(output_dir.glob("*.json")):
            try:
                tasks.append(self.task_class.model_validate_json(path.read_text()))
            except ValidationError:
                logger.warning(
                    "ignoring {}: does not validate as {}",
                    path.name,
                    self.task_class.__name__,
                )
                continue
        return tasks
