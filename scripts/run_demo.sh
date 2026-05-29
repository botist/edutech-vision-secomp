#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

"$ROOT/run.sh" --setup-only
exec "$ROOT/.venv/bin/python" -m edutech_vision --mode demo --showcase --fullscreen "$@"
