#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODE="${1:---local}"

print_usage() {
  cat <<'EOF'
Usage:
  ./run.sh --local    # setup venv, install deps, run tests, run demo (default)
  ./run.sh --docker   # build image and run demo in Docker
  ./run.sh --compose  # run with Docker Compose
EOF
}

require_openai_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ ! -f ".env" ]] || ! grep -qE '^OPENAI_API_KEY=.+$' .env; then
      echo "Error: OPENAI_API_KEY is not set and was not found in .env" >&2
      echo "Set it with: export OPENAI_API_KEY=... or add OPENAI_API_KEY=... to .env" >&2
      exit 1
    fi
  fi
}

run_local() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required but not found." >&2
    exit 1
  fi

  if [[ ! -d ".venv" ]]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .venv
  fi

  local venv_py=".venv/bin/python"
  local venv_pip=".venv/bin/pip"

  if [[ ! -x "$venv_py" ]]; then
    echo "Error: virtual environment is missing Python at $venv_py" >&2
    exit 1
  fi

  "$venv_py" -m pip install --upgrade pip >/dev/null

  echo "[setup] Installing dependencies..."
  "$venv_pip" install -r requirements.txt >/dev/null

  require_openai_key

  echo "[check] Running smoke tests..."
  "$venv_py" -m unittest -q

  echo "[run] Starting Self-RAG demo..."
  "$venv_py" SELFRAG-agent.py
}

run_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is required but not found." >&2
    exit 1
  fi

  require_openai_key

  echo "[docker] Building image..."
  if docker build --ignorefile docker/.dockerignore -f docker/Dockerfile -t self-rag-from-scratch:latest .; then
    echo "[docker] Running container..."
    docker run --rm --env-file .env self-rag-from-scratch:latest
    return
  fi

  echo "[docker] Image build failed on this host. Falling back to no-build container run..."
  docker run --rm \
    --env-file .env \
    -e USER_AGENT=self-rag-from-scratch/1.0 \
    -v "$PROJECT_DIR:/app" \
    -w /app \
    python:3.12-slim \
    bash -lc "pip install -q --no-cache-dir -r requirements.txt && python SELFRAG-agent.py"
}

run_compose() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is required but not found." >&2
    exit 1
  fi

  require_openai_key

  echo "[compose] Starting service..."
  docker compose -f docker/docker-compose.yml up --build
}

case "$MODE" in
  --local)
    run_local
    ;;
  --docker)
    run_docker
    ;;
  --compose)
    run_compose
    ;;
  -h|--help)
    print_usage
    ;;
  *)
    echo "Error: unknown mode '$MODE'" >&2
    print_usage
    exit 1
    ;;
esac
