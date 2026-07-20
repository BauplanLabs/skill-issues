CLI_TO_SDK: dict[str, list[str]] = {
    "bauplan branch create": ["create_branch"],
    "bauplan branch merge": ["merge_branch"],
    "bauplan info": ["info"],
    "bauplan query": [
        "query",
        "query_to_csv_file",
        "query_to_generator",
        "query_to_json_file",
        "query_to_parquet_file",
    ],
    "bauplan run": ["run"],
    "bauplan table create": [
        "create_table",
        "plan_table_creation",
        "apply_table_creation_plan",
        "create_external_table_from_parquet",
        "create_external_table_from_metadata",
    ],
    "bauplan table import": ["import_data"],
}


def sdk_methods_for(cmd: str) -> list[str]:
    """Return the bauplan SDK method names equivalent to a CLI command.

    `expected_commands` entries are normalized command phrases (e.g.
    "bauplan branch create"). Match the exact phrase, falling back to the
    longest known prefix so trailing args still resolve. Empty list means the
    command has no SDK equivalent
    """
    cmd = cmd.strip()
    if cmd in CLI_TO_SDK:
        return CLI_TO_SDK[cmd]
    for key in sorted(CLI_TO_SDK, key=len, reverse=True):
        if cmd.startswith(key):
            return CLI_TO_SDK[key]
    return []
