# GEPA reflection prompt for optimizing Bauplan SKILL.md candidates.
# These strings reach `gepa.optimize_anything(..., objective=..., background=...)`
# via the DomainSpec in domain.py; GEPA renders them into its reflection prompt
# as the "Optimization Goal" and "Domain Context & Constraints" sections. The
# reflection LM that rewrites the skill cannot make tool calls or fetch URLs,
# so any Bauplan command it emits comes from memory, which is why it
# hallucinates syntax. BACKGROUND gives it an authoritative, hand-synthesized
# reference to copy from instead, plus guardrails against overfitting the skill
# to the evaluation harness. Keep it in sync with the real skills under
# `harness/tasks/bauplan-eval/environment/skills/` in this domain; the command
# reference below is distilled from those (verified against the live CLI/SDK),
# not invented

OBJECTIVE = (
    "Improve this Bauplan SKILL.md so a general coding agent solves data "
    "engineering tasks on a Bauplan lakehouse: invoke the right skills and "
    "commands, satisfy the state and response checks, and avoid forbidden "
    "actions. The output is a reusable, general-purpose skill that ships to "
    "real users: write durable domain guidance, not a solution tuned to any "
    "one task or scoring setup."
)

# This is the background context for the GEPA reflection.
# You can add here additional instructions, constraints, or references that the reflection LM should consider
# when rewriting the skill.
BACKGROUND = """\
## What you are writing

You are rewriting a general-purpose Bauplan skill (a `SKILL.md`) that ships to
real users and is loaded by an agent facing a wide range of unseen tasks. The
evaluation feedback you see is a *sample* of tasks, not the target. Optimize the
skill for the whole domain, not for the specific examples in the feedback.

FILL HERE WITH YOUR BACKGROUND INSTRUCTIONS...
"""
