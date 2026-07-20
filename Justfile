root := justfile_directory()

_list:
    @just --list --unsorted

# Install dependencies and bootstrap .env
[group("installation")]
install:
    @cd {{root}} && uv sync --all-packages
    @cd {{root}} && [ -f .env ] || { cp .env.example .env && echo "Created .env from .env.example: fill in the blank values."; }

# Scaffold a new domain
[group("installation")]
new-domain name:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{root}}
    name="{{name}}"
    # Double underscores are reserved for the per-user worker copies (gitignored)
    if ! [[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || [[ "$name" == *"__"* ]]; then
      echo "invalid domain name '$name': use a lowercase Python identifier without double underscores" >&2
      exit 1
    fi
    if [ -e "src/domains/$name" ]; then
      echo "src/domains/$name already exists" >&2
      exit 1
    fi
    cp -R src/domains/_template "src/domains/$name"
    echo "Created src/domains/$name. Next steps:"
    echo "  1. fill in src/domains/$name/config.yaml (models, paths, upstream skill URLs)"
    echo "  2. write the prompts in src/domains/$name/prompts/"
    echo "  3. define the task model in schema.py and the generator in generate.py"
    echo "  4. point harness/job.yaml at your Modal secret and flesh out harness/tasks/"

# Lint and format the codebase
[group("code-quality")]
lint:
    @cd {{root}} && uvx ruff format && uvx ruff check --fix

# Type-check the codebase
[group("code-quality")]
type-check:
    @cd {{root}} && uv run ty check .

# Run all code-quality checks
[group("code-quality")]
check: lint type-check

# Shared launcher: require and source the root .env, then invoke src.main
_main +args:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{root}}
    if [ ! -f .env ]; then
      echo "Missing {{root}}/.env: run 'just install' or copy .env.example to .env and fill it in" >&2
      exit 1
    fi
    set -a
    . .env
    set +a
    uv run python -m src.main {{args}}

# Generate benchmark tasks for a domain
[group("dataset")]
generate *args:
    @(just _main "generate" {{args}})

# Optimize a skill's SKILL.md via GEPA
[group("optimization")]
optimize *args:
    @(just _main "optimize" {{args}})

# Prove the eval is real on one dataset entry
[group("optimization")]
canary *args:
    @(just _main "canary" {{args}})

# Restore every local SKILL.md to its upstream version
[group("skills")]
reset-skills *args:
    @(just _main "reset-skills" {{args}})

# Diff a local SKILL.md against its upstream version
[group("skills")]
skill-diff *args:
    @(just _main "skill-diff" {{args}})

# Launch one detached worker per <skill>:<user> pair
[group("optimization")]
launch-parallel domain base_user +pairs:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{root}}"
    cd "$root"
    n_calls="${N:-15}"
    harbor_bin_dir="${HARBOR_BIN_DIR:-$HOME/.local/bin}"
    domain_env="$root/src/domains/{{domain}}/.env"
    mkdir -p "$root/jobs"

    for pair in {{pairs}}; do
      skill="${pair%%:*}"
      user="${pair##*:}"
      worker_domain="{{domain}}__${user}"
      src_dir="$root/src/domains/{{domain}}"
      dst_dir="$root/src/domains/${worker_domain}"

      # Fresh isolated copy: drop prior run artifacts so each worker starts clean
      rm -rf "$dst_dir"
      rsync -a --exclude 'harness/jobs/' --exclude 'telemetry/' --exclude '__pycache__/' \
        "$src_dir/" "$dst_dir/"
      mkdir -p "$dst_dir/harness/jobs"

      sed -i.bak "s/{{base_user}}/${user}/g" "$dst_dir/harness/job.yaml"
      rm -f "$dst_dir/harness/job.yaml.bak"

      ts="$(date +%Y%m%d-%H%M%S)"
      log="$root/jobs/optimize-${skill}-${user}-${ts}.log"
      nohup bash -c "
        cd '$root'
        set -a; . ./.env; set +a
        [ -f '$domain_env' ] && { set -a; . '$domain_env'; set +a; }
        . .venv/bin/activate
        export PATH=\"${harbor_bin_dir}:\$PATH\"
        exec python -m src.main optimize '${worker_domain}' '${skill}' --n ${n_calls}
      " > "$log" 2>&1 < /dev/null &
      echo "launched skill '${skill}' as worker '${user}' -> domain ${worker_domain}"
      echo "  log: ${log}"
      sleep 2
    done

    echo
    echo "detached workers:"
    ps -eo pid,ppid,etime,command | grep "src.main optimize" | grep -v grep || true

# Watch one optimize worker
[group("optimization")]
watch pattern jobs_dir label="":
    #!/usr/bin/env bash
    set -uo pipefail
    root="{{root}}"
    cd "$root"
    jobs_dir="{{jobs_dir}}"
    label="{{label}}"
    [ -n "$label" ] || label="$(basename "$(dirname "$jobs_dir")")"

    render() { uv run python -m src.core.plot_curve --jobs-dir "$jobs_dir" >/dev/null 2>&1 || true; }

    pid="$(pgrep -f -- "{{pattern}}" | head -1 || true)"
    echo "[$label] watching '{{pattern}}' (pid ${pid:-none}); jobs $jobs_dir"
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
      render
      sleep 90
      pid="$(pgrep -f -- "{{pattern}}" | head -1 || true)"
    done
    render
    png="$(ls -t "$jobs_dir"/gepa-*/*-curve.png 2>/dev/null | head -1 || true)"
    echo "[$label] run ENDED. final plot: ${png:-<none>}"
