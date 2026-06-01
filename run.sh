#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SETUP_ONLY=0
NO_LAUNCH=0
SKIP_MODEL=0
PYTHON_REQUEST="3.11"
TOOLS_DIR="$ROOT/.tools"
UV_DIR="$TOOLS_DIR/uv"
UV_PYTHON_DIR="$TOOLS_DIR/uv-python"
UV_BIN="$UV_DIR/uv"
VENV_DIR="$ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
STAMP_PATH="$VENV_DIR/.edutech-bootstrap"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only|-SetupOnly)
      SETUP_ONLY=1
      ;;
    --no-launch|-NoLaunch)
      NO_LAUNCH=1
      ;;
    --skip-model|-SkipModel)
      SKIP_MODEL=1
      ;;
    -h|--help)
      cat <<'EOF'
Uso: ./run.sh [--setup-only] [--no-launch] [--skip-model]

Prepara o ambiente local e abre o EduTech Vision Control Center.
EOF
      exit 0
      ;;
    *)
      echo "Argumento desconhecido: $1" >&2
      exit 2
      ;;
  esac
  shift
done

step() {
  printf '\n== %s ==\n' "$1"
}

die() {
  echo "ERRO: $1" >&2
  exit 1
}

assert_unix_desktop() {
  case "$(uname -s)" in
    Linux|Darwin) ;;
    *) die "Use run.bat/run.ps1 no Windows; run.sh e para Linux/macOS." ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64|arm64|aarch64) ;;
    *) die "Sistema 64-bit e necessario para os wheels do MediaPipe/OpenCV." ;;
  esac
}

invoke_python_probe() {
  local python_exe="$1"
  "$python_exe" - <<'PY' >/dev/null 2>&1
import platform
import sys

ok = sys.version_info[:2] == (3, 11) and platform.architecture()[0] == "64bit"
raise SystemExit(0 if ok else 1)
PY
}

find_compatible_python() {
  local candidate
  for candidate in "$VENV_PYTHON" python3.11 python3 python; do
    if [[ "$candidate" == "$VENV_PYTHON" ]]; then
      [[ -x "$candidate" ]] || continue
    elif ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi

    if invoke_python_probe "$candidate"; then
      command -v "$candidate" 2>/dev/null || printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  if [[ -x "$UV_BIN" ]]; then
    printf '%s\n' "$UV_BIN"
    return 0
  fi

  step "Baixando uv local"
  mkdir -p "$UV_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
  else
    die "Instale curl/wget ou Python 3.11 antes de rodar o setup."
  fi

  [[ -x "$UV_BIN" ]] || die "uv nao foi encontrado em $UV_BIN."
  printf '%s\n' "$UV_BIN"
}

create_venv_with_python() {
  local python_exe="$1"
  step "Criando ambiente virtual .venv"
  if "$python_exe" -m venv "$VENV_DIR"; then
    return 0
  fi
  return 1
}

create_venv_with_uv() {
  local uv_exe
  uv_exe="$(install_uv)"
  step "Criando ambiente virtual .venv com uv"
  mkdir -p "$UV_PYTHON_DIR"
  export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR"
  "$uv_exe" python install "$PYTHON_REQUEST"
  "$uv_exe" venv --seed --python "$PYTHON_REQUEST" "$VENV_DIR"
}

ensure_venv() {
  if [[ -x "$VENV_PYTHON" ]] && invoke_python_probe "$VENV_PYTHON"; then
    return
  fi

  if [[ -d "$VENV_DIR" ]]; then
    local backup
    backup="$ROOT/.venv.backup-$(date +%Y%m%d-%H%M%S)"
    step "Arquivando .venv incompativel"
    mv "$VENV_DIR" "$backup"
  fi

  local python_exe=""
  if python_exe="$(find_compatible_python)"; then
    if create_venv_with_python "$python_exe"; then
      return
    fi
    rm -rf "$VENV_DIR"
    echo "Python encontrado, mas modulo venv/ensurepip falhou; usando uv como fallback."
  fi

  create_venv_with_uv
}

file_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print toupper($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print toupper($1)}'
  else
    "$VENV_PYTHON" - "$path" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest().upper())
PY
  fi
}

setup_stamp() {
  local file
  for file in requirements.txt pyproject.toml scripts/download_models.py; do
    [[ -f "$file" ]] && file_sha256 "$file"
  done | paste -sd "|"
}

ensure_dependencies() {
  [[ -x "$VENV_PYTHON" ]] || die "Python do ambiente .venv nao encontrado."

  local stamp current
  stamp="$(setup_stamp)"
  current=""
  [[ -f "$STAMP_PATH" ]] && current="$(cat "$STAMP_PATH")"
  if [[ "${current//$'\n'/}" == "$stamp" ]]; then
    return
  fi

  step "Instalando dependencias Python"
  "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$VENV_PYTHON" -m pip install --upgrade pip --no-warn-script-location
  "$VENV_PYTHON" -m pip install --no-warn-script-location -r requirements.txt
  "$VENV_PYTHON" -m pip install --no-warn-script-location -e .
  printf '%s\n' "$stamp" > "$STAMP_PATH"
}

ensure_model() {
  [[ "$SKIP_MODEL" -eq 1 ]] && return
  step "Verificando modelos MediaPipe e YuNet"
  "$VENV_PYTHON" scripts/download_models.py
}

check_launcher_ready() {
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import tkinter  # noqa: F401
import edutech_vision.launcher  # noqa: F401
PY
}

print_unix_gui_help() {
  case "$(uname -s)" in
    Linux)
      cat <<'EOF'

Setup concluido, mas o Control Center precisa de Tkinter/bibliotecas de desktop.
Em Ubuntu/Debian, normalmente resolve com:
  sudo apt install python3-tk libgl1 libglib2.0-0

Enquanto isso, o CLI ja esta instalado. Exemplo:
  ./.venv/bin/python -m edutech_vision --mode demo --showcase
EOF
      ;;
    Darwin)
      cat <<'EOF'

Setup concluido, mas o Control Center precisa de Tkinter/Tcl-Tk.
No macOS, use um Python 3.11 com Tkinter funcional, por exemplo via python.org ou Homebrew.

Enquanto isso, o CLI ja esta instalado. Exemplo:
  ./.venv/bin/python -m edutech_vision --mode demo --showcase
EOF
      ;;
  esac
}

assert_unix_desktop
ensure_venv
ensure_dependencies
ensure_model

if [[ "$SETUP_ONLY" -eq 1 || "$NO_LAUNCH" -eq 1 ]]; then
  printf '\nSetup concluido.\n'
  exit 0
fi

step "Abrindo EduTech Vision Control Center"
if ! check_launcher_ready; then
  print_unix_gui_help
  exit 1
fi

exec "$VENV_PYTHON" -m edutech_vision.launcher
