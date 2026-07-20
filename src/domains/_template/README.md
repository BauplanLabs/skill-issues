# Domain template

Copyable skeleton for a new skill-optimization domain. It type-checks and imports as-is; every placeholder is written in `<angle brackets>`. Create your domain with `just new-domain <name>`, which copies this package to `src/domains/<name>/`, then fill the placeholders in the order below. Domain-internal imports here are package-relative (`from .schema import ...`, `from .prompts.scenario import ...`), so the copy is self-contained and needs no import rewriting; imports from `src.core` stay absolute.

## What to fill in

`config.yaml` first. Point `upstream_skills_raw`, `upstream_skills_api`, and `skills_catalog_url` at your skills repository (`skills_catalog_url` may also be a local path). Set `models.agent` / `models.judge` to harness-native ids and `models.generator` to a full litellm id. The `paths` values are resolved relative to the package directory; if you rename the `template-eval` harness task directory, update the `skills` and `dataset_entry` paths to match.

`schema.py` next. Replace the placeholder `Category.EXAMPLE` with the real task categories, and add any backend-specific fields to `TemplateBenchmarkTask` (an initial-state model, a narrowed `StateAssertion` Literal). If your harness injects globals into `validation_script`, define a stub of their signatures and thread it into `validate_task`.

`prompts/scenario.py` and `prompts/data_entry.py`. `SCENARIO_PROMPT` is the per-iteration user message; `DATA_ENTRY_SYSTEM_PROMPT` is the system prompt. Fill the labelled sections: what the platform is, the grounding data that exists, the per-category intents, the prompt-versus-grading rules, and the constraint-authoring rules. The `{{skills}}`, `{{existing}}`, `{{categories}}`, and `{{specificities}}` placeholders are substituted by `generate.py`; leave them in place.

`prompts/gepa_reflection.py` and `prompts/review.py`. `OBJECTIVE` and `BACKGROUND` feed GEPA's reflection prompt (the reflection LM cannot fetch anything, so `BACKGROUND` must carry the authoritative CLI/SDK reference). `REVIEW_GUIDANCE` is appended to the core semantic-review prompt; add the protected invariants and category-specific hard rules, or leave it empty.

`domain.py` and `generate.py` are mostly wired already. `domain.py` assembles `SPEC` from the prompt strings; set `protected_assertions` if your domain has standing invariants. In `generate.py`, extend `build_system_prompt` with any extra grounding your `DATA_ENTRY_SYSTEM_PROMPT` needs, and replace `live_check` if you have a live backend to gate against.

`harness/` last. In `job.yaml`, set the Modal secret and `app_name` (the `<base-user>` substring is the rewrite anchor `just launch-parallel` swaps per worker). In `harness/tasks/template-eval/`, fill `task.toml` (identity, timeouts, env pass-through), add your runtime dependencies to `environment/Dockerfile`, drop your real skills into `environment/skills/`, implement the initial-state hook in `solution/solve.sh`, and write the real checks in `tests/test.sh` (it must produce `/logs/verifier/reward.txt` as a single float and a human-readable `pytest.log`).
