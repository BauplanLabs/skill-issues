from typing import Any, cast

import litellm


def call_tool(
    model: str,
    system_text: str,
    user_msg: str,
    tool_schema: dict[str, Any],
    max_tokens: int | None = None,
) -> str | None:
    """Force a single tool call named `tool_schema["name"]` and return its JSON args.

    `model` is a full litellm model id (provider prefix included), so routing
    is entirely litellm's concern. litellm normalizes every provider response
    to a single tool-call shape whose `function.arguments` is a JSON string.
    Returns None when the model emitted no matching tool call. When
    `max_tokens` is None, litellm fills any provider-required default itself.
    """
    tool_name = tool_schema["name"]
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_msg},
        ],
        tools=[{"type": "function", "function": tool_schema}],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        max_tokens=max_tokens,
    )
    message = cast(Any, response).choices[0].message
    for tc in message.tool_calls or []:
        if tc.function.name == tool_name:
            return tc.function.arguments
    return None
