import contextlib
import io
import urllib.request
from pathlib import Path
from typing import cast

import yaml

from src.core.domain import DomainSpec
from src.core.generate import BenchmarkGenerator
from src.core.schema import BenchmarkTask, PromptSpecificity
from src.core.utils import render
from .domain import SPEC
from .prompts.data_entry import DATA_ENTRY_SYSTEM_PROMPT
from .prompts.scenario import SCENARIO_PROMPT
from .schema import (
    BauplanBenchmarkTask,
    BauplanStateAssertion,
    TaskCategory,
    seed_catalog,
    validate_task,
)

_CONFIG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


def _fetch_lakehouse_snapshot(ref: str = "main") -> str:
    """Return a textual snapshot of the SEEDED tables on `ref` with column types.

    Used to ground the generator so it cannot hallucinate table or column
    names. One line per table: `- ns.tbl: col1: type1, ...`.
    Uses `get_table` per table because `get_tables` does not populate
    `.fields` reliably (the established pattern in `verify_bauplan_state.py`).

    Only SEEDED tables (guaranteed-permanent, listed in datasets.json) are
    shown. A task may only require a seeded table to pre-exist, so ephemeral
    leftovers on `ref` (prior pipeline outputs, ingestion residue) are omitted
    entirely; listing them only tempted the generator into tasks that then fail
    validation for depending on a non-seeded table.
    """
    import bauplan

    from .schema import seeded_tables

    seeded = seeded_tables()
    client = bauplan.Client()
    lines: list[str] = []
    for tbl in sorted(client.get_tables(ref=ref), key=lambda t: t.fqn):
        if tbl.fqn not in seeded:
            continue
        meta = client.get_table(table=tbl.fqn, ref=ref)
        cols = ", ".join(f"{f.name}: {f.type}" for f in meta.fields)
        body = cols or "(no columns)"
        lines.append(f"- {tbl.fqn}: {body}")
    header = (
        "Every table below is SEEDED (permanent, listed in datasets.json) and may "
        "be required to pre-exist. Any data NOT listed here is not guaranteed "
        "present; a task needing it must ingest a listed S3 seed file itself.\n"
    )
    return header + "\n".join(lines) if lines else "(no seeded tables on `main`)"


class Generator(BenchmarkGenerator):
    @property
    def task_class(self) -> type[BauplanBenchmarkTask]:
        return BauplanBenchmarkTask

    @property
    def spec(self) -> DomainSpec:
        """The lakehouse domain's reflection prompts and review policy."""
        return SPEC

    @property
    def skills_catalog_url(self) -> str:
        return _CONFIG["skills_catalog_url"]

    @property
    def prompt(self) -> str:
        return SCENARIO_PROMPT

    def all_categories(self) -> list[TaskCategory]:
        return list(TaskCategory)

    def parse_category_specs(self, specs: list[str]) -> list[TaskCategory]:
        """Resolve --category strings to TaskCategory values.

        Accepts literal category values (e.g. `read_explore`,
        `write_ingest`) and the group prefixes `read` / `write`, which
        expand to every category whose `.value` starts with that prefix
        followed by `_`. Duplicates are deduplicated while preserving
        first-seen order.
        """
        by_name = {c.value: c for c in TaskCategory}
        out: list[TaskCategory] = []
        for spec in specs:
            if spec in by_name:
                out.append(by_name[spec])
                continue
            group = [c for c in TaskCategory if c.value.startswith(f"{spec}_")]
            if group:
                out.extend(group)
                continue
            raise ValueError(
                f"unknown category {spec!r}; valid: {sorted(by_name)} "
                f"or groups {{'read', 'write'}}"
            )
        seen: set[TaskCategory] = set()
        deduped: list[TaskCategory] = []
        for c in out:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    def build_system_prompt(
        self,
        existing: list[BenchmarkTask],
        categories: list[TaskCategory],
        skill_catalog: list[str],
        specificities: list[PromptSpecificity],
    ) -> str:
        reference_sections: list[str] = []
        for label, url in _CONFIG["reference_urls"].items():
            with urllib.request.urlopen(url) as resp:
                body = resp.read().decode("utf-8")
            reference_sections.append(
                f"--- {label.upper()} reference ({url}) ---\n{body}"
            )
        reference = "\n\n".join(reference_sections)
        catalog_lines = [
            f"- {name} ({d.format}, {d.files} file(s), "
            f"S3 URI: {d.search_uri}): {d.description}"
            for name, d in seed_catalog().items()
        ]
        bauplan_existing = cast(list[BauplanBenchmarkTask], existing)
        existing_lines = [
            f"- {t.task_name} ({t.category.value}): {t.goal}" for t in bauplan_existing
        ] or ["- (none yet)"]
        skill_lines = [f"- {s}" for s in skill_catalog] or ["- (none)"]
        lakehouse = _fetch_lakehouse_snapshot("main")
        return render(
            DATA_ENTRY_SYSTEM_PROMPT,
            catalog="\n".join(catalog_lines),
            skills="\n".join(skill_lines),
            reference=reference,
            existing="\n".join(existing_lines),
            categories=", ".join(c.value for c in categories),
            specificities=", ".join(s.value for s in specificities),
            lakehouse=lakehouse,
        )

    def validate(self, task: BenchmarkTask) -> bool:
        self._repair_no_change_main(cast(BauplanBenchmarkTask, task))
        return validate_task(cast(BauplanBenchmarkTask, task))

    @staticmethod
    def _repair_no_change_main(task: BauplanBenchmarkTask) -> None:
        """Inject the always-required `no_change(main)` assertion when the model
        omits it, rather than rejecting the whole task.

        Global `main` is read-only for every task (agents publish only to their
        personal `<username>.main`), so this invariant is universal and always
        safe to add. Auto-repairing it (instead of a hard reject) removes the
        dominant read-task reject cause and roughly doubles generation throughput.
        Idempotent: only appended when absent. Mutates in place, so the repaired
        assertion is what gets written to disk.
        """
        sas = task.end_conditions.state_assertions

        def _has(target: str) -> bool:
            return any(sa.kind == "no_change" and sa.target == target for sa in sas)

        if not _has("main"):
            sas.append(
                BauplanStateAssertion(
                    kind="no_change",
                    target="main",
                    expected=None,
                    description=(
                        "The shared global `main` branch must remain unchanged; "
                        "agents publish only to their personal `<username>.main`."
                    ),
                )
            )
        # Read-only tasks write nothing, so the agent's personal main must also
        # stay unchanged (documented read-task invariant). Gated to read_* only:
        # write tasks legitimately mutate <username>.main when publishing.
        if task.category.value.startswith("read_") and not _has("<username>.main"):
            sas.append(
                BauplanStateAssertion(
                    kind="no_change",
                    target="<username>.main",
                    expected=None,
                    description=(
                        "A read-only task writes nothing; the agent's personal "
                        "`<username>.main` branch must remain unchanged."
                    ),
                )
            )

    def live_check(self, task: BenchmarkTask) -> tuple[bool, str]:
        """For read-only tasks, run the validation_script against clean state.

        A `read_*` task is read-only: no agent action is required, so its
        `validation_script` must already pass against the live lakehouse as-is.
        Executing it here rejects entries whose verification is unsolvable
        (false data assumptions, e.g. an all-NULL column) or buggy (malformed or
        engine-unsupported SQL, over-broad guards that trip on pre-existing
        state) before they reach the dataset. Write tasks are skipped: their
        clean-state validation is expected to fail because the agent's output
        does not exist yet.
        """
        bt = cast(BauplanBenchmarkTask, task)
        category = getattr(bt.category, "value", str(bt.category))
        if not category.startswith("read"):
            return True, ""
        script = (bt.validation_script or "").strip()
        if not script:
            return False, "read task has empty validation_script"

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ns: dict = {"__name__": "__main__"}
                try:
                    exec(compile(script, "<validation_script>", "exec"), ns)
                    code: object = 0
                except SystemExit as exc:
                    code = exc.code or 0
        except Exception as exc:  # any failure means the script can't pass clean
            tail = (buf.getvalue().strip().splitlines() or [""])[-1]
            return False, f"{type(exc).__name__}: {exc} :: {tail}"

        if code != 0:
            tail = (buf.getvalue().strip().splitlines() or [""])[-1]
            return False, f"validation_script exited {code!r} :: {tail}"
        return True, ""
