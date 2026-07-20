# System prompt for the data-entry LM that emits one BenchmarkTask per call
# (rendered in generate.py via `render(DATA_ENTRY_SYSTEM_PROMPT, ...)`).
#
# THIS IS A TEMPLATE. The skeleton below shows the SHAPE of the prompt: the
# sections it needs and the `{{...}}` placeholders that generate.py fills in at
# runtime (catalog, lakehouse, skills, reference, existing, categories,
# specificities). Fill in each
# `FILL HERE ...` block with your own domain constraints.
DATA_ENTRY_SYSTEM_PROMPT = (
    "Emit exactly one new BenchmarkTask via the `emit_task` tool, "
    "conforming to the schema and the constraints below.\n\n"
    "AVAILABLE SEED DATASETS: these source files live in public S3 "
    "buckets (each entry below shows its `s3://` URI). They are NOT "
    "pre-loaded as tables in the lakehouse, and nothing materializes "
    "them automatically before the agent runs:\n"
    "{{catalog}}"
    "\n\nLIVE LAKEHOUSE SNAPSHOT on global `main` (captured at generation "
    "time); these are the tables that actually exist right now, one line "
    "per table as `- namespace.table: col: type, col: type, ...`:\n"
    "{{lakehouse}}"
    "\n\nAVAILABLE BAUPLAN SKILLS (reference by name in `expected_skills`):\n"
    "{{skills}}"
    "\n\nBAUPLAN REFERENCE (authoritative; contains both the CLI reference "
    "and the Python SDK reference):\n"
    "{{reference}}"
    "\n\nEXISTING TASKS (cover new ground, do not duplicate these):\n"
    "{{existing}}"
    "\n\nAllowed categories for this run: {{categories}}"
    "\n\nAllowed prompt specificities for this run: {{specificities}}"
    "\n\nFILL HERE YOUR INITIAL-STATE RULES: which preconditions "
    "(`expected_tables`, `external_datasets`, `absent_tables`) each kind of "
    "task must declare, and how they must be grounded in the catalog and "
    "snapshot above so the harness can verify the lakehouse BEFORE the agent "
    "runs...\n\n"
    "FILL HERE YOUR BRANCHING CONVENTION: the branch layout every generated "
    "task must assume, what a write task is expected to do, and which parts of "
    "that workflow the prompt may state versus which the agent is expected to "
    "know from its skills...\n\n"
    "FILL HERE YOUR CATEGORY INTENTS: for each allowed category, what the task "
    "asks the agent to do, and which bauplan commands a correct solution "
    "genuinely invokes (`expected_commands`)...\n\n"
    "FILL HERE YOUR GRADING RULES: when hard versus soft response constraints "
    "are required, which `state_assertions` kinds and targets are valid, which "
    "`forbidden_actions` each category must carry, and what the "
    "`validation_script` must check (it runs after the task, exits 0 on "
    "success and non-zero on failure)...\n\n"
    "FILL HERE YOUR PROMPT-SPECIFICITY DEFINITIONS: what `vague`, `medium`, "
    "and `detailed` mean as user personas, and how the `prompt` field must be "
    "written to match the level you tag...\n"
)
