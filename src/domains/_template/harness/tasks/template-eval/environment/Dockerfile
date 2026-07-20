FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN uv init --bare . \
 && uv add claude-agent-sdk
# add this domain's runtime dependencies here, e.g. `uv add <your-backend-sdk>`

# Skills baked into the image. The optimizer rewrites the host SKILL.md (this build
# context) per candidate; this COPY layer is content-hashed, so a changed skill busts
# the cache and the rebuilt image carries the candidate. task.toml's skills_dir then
# points harbor at this path to load skills into the agent.
COPY skills/ /app/.claude/skills/

# Keep the Modal sandbox alive: harbor creates the sandbox with no command and
# execs the agent/verifier into it, so the image needs a long-running entrypoint
CMD ["sleep", "infinity"]
