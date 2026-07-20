# OBJECTIVE and BACKGROUND feed GEPA's reflection prompt via the DomainSpec in
# domain.py. The reflection LM that rewrites the skill cannot make tool calls or
# fetch URLs, so BACKGROUND must carry an authoritative reference for it to copy
# from instead of guessing. Keep it in sync with the real skills under
# harness/tasks/template-eval/environment/skills/

OBJECTIVE = (
    "Improve this <platform> SKILL.md so a general coding agent <accomplishes "
    "the domain's core tasks>: invoke the right skills and commands, satisfy "
    "the state and response checks, and avoid forbidden actions. The output is "
    "a reusable, general-purpose skill that ships to real users: write durable "
    "domain guidance, not a solution tuned to any one task or scoring setup."
)

BACKGROUND = """\
## What you are writing

<One paragraph: you are rewriting a general-purpose <platform> skill loaded by
an agent facing a wide range of unseen tasks. The evaluation feedback you see is
a sample, not the target. Optimize for the whole domain and never overfit to the
grader, hardcode task-specific answers, or name specific tasks.>

## <Platform> reference (copy this syntax; do not invent commands)

<You cannot run commands or fetch docs, so paste the authoritative surface the
skill should teach from. Include:>

- Platform conventions: <naming, state/branching model, and the safety
  invariants the skill must always uphold>.
- CLI / SDK surface: <the exact commands and method signatures in scope,
  verified against the real tool, not recalled from memory>.
- Invariants the skill must teach: <what a correct workflow always does and
  never does, phrased as durable best practice>.
"""
