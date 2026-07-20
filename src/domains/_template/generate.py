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
from .schema import Category, TemplateBenchmarkTask, validate_task

_CONFIG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


class Generator(BenchmarkGenerator):
    @property
    def task_class(self) -> type[TemplateBenchmarkTask]:
        return TemplateBenchmarkTask

    @property
    def spec(self) -> DomainSpec:
        return SPEC

    @property
    def skills_catalog_url(self) -> str:
        return _CONFIG["skills_catalog_url"]

    @property
    def prompt(self) -> str:
        return SCENARIO_PROMPT

    def all_categories(self) -> list[Category]:
        return list(Category)

    def build_system_prompt(
        self,
        existing: list[BenchmarkTask],
        categories: list[Category],
        skill_catalog: list[str],
        specificities: list[PromptSpecificity],
    ) -> str:
        skill_lines = [f"- {s}" for s in skill_catalog] or ["- (none)"]
        existing_lines = [f"- {t.task_name}" for t in existing] or ["- (none yet)"]
        return render(
            DATA_ENTRY_SYSTEM_PROMPT,
            skills="\n".join(skill_lines),
            existing="\n".join(existing_lines),
            categories=", ".join(c.value for c in categories),
            specificities=", ".join(s.value for s in specificities),
        )

    def validate(self, task: BenchmarkTask) -> bool:
        return validate_task(cast(TemplateBenchmarkTask, task))

    def live_check(self, task: BenchmarkTask) -> tuple[bool, str]:
        """Accept every entry: this template ships with no live backend to gate against.

        A domain with a real backend rejects entries whose verification cannot
        pass here; see src/domains/bauplan_lakehouse/generate.py live_check,
        which executes read-task validation_scripts against the live lakehouse.
        """
        return True, ""
