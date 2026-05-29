#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python local nao encontrado. Rode ./run.sh primeiro." >&2
  exit 1
fi

exec "$PYTHON" "$@"
