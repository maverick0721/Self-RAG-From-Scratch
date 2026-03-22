#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but not found." >&2
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv .venv
fi

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: virtual environment is missing Python at $VENV_PY" >&2
  exit 1
fi

# Keep pip quiet and deterministic for demo setup.
"$VENV_PY" -m pip install --upgrade pip >/dev/null

echo "[setup] Installing dependencies..."
"$VENV_PIP" install -r requirements.txt >/dev/null

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ ! -f ".env" ]] || ! grep -qE '^OPENAI_API_KEY=.+$' .env; then
    echo "Error: OPENAI_API_KEY is not set and was not found in .env" >&2
    echo "Set it with: export OPENAI_API_KEY=... or add OPENAI_API_KEY=... to .env" >&2
    exit 1
  fi
fi

echo "[check] Running smoke tests..."
"$VENV_PY" -m unittest -q

echo "[run] Starting Self-RAG demo..."
"$VENV_PY" SELFRAG-agent.py
