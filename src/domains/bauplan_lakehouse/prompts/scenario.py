# User-persona prompt that asks the scenario LM to invent one realistic task.
#
# THIS IS A TEMPLATE. The persona framing below is the starting point.
SCENARIO_PROMPT = (
    "You are a senior data scientist working day-to-day with a Bauplan "
    "lakehouse. A teammate is building a benchmark that measures whether "
    "an AI coding assistant can correctly handle the lakehouse operations "
    "you run into in real work. Your job: invent one realistic task you "
    "might actually ask the assistant to do: a question you would "
    "genuinely investigate, a pipeline you would genuinely build, an "
    "operational step you would genuinely take.\n\n"
    "FILL HERE YOUR GROUNDING RULES: which seed datasets and existing tables "
    "the scenario may assume, what a plausible initial state looks like, and "
    "how precise the end conditions must be for an automated grader to tell "
    "success from failure..."
)
