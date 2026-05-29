#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

"$ROOT/run.sh" --setup-only
exec "$ROOT/.venv/bin/python" -m edutech_vision --mode plateia --showcase --fullscreen --max-faces 24 --face-confidence 0.60 --audience-yaw-tolerance 30 --audience-pitch-tolerance 20 "$@"
