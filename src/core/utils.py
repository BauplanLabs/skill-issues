def render(template: str, **values: str) -> str:
    """Substitute {{key}} placeholders in `template` with the given values.

    Used instead of str.format so the template can freely contain single
    { / } (e.g., regexes, JSON snippets) without escaping.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return out
