#!/usr/bin/env bash
# Run the Comparator check for the paper-facing challenge/solution pair.
#
# By default this script requires a real `landrun` executable.  Local macOS
# development can opt into Comparator's unsandboxed fake landrun shim by setting
# `COMPARATOR_ALLOW_FAKE_LANDRUN=1`; that mode is only a smoke test.

set -euo pipefail

cd "$(dirname "$0")/.."

LEAN428_BIN="${LEAN428_BIN:-$HOME/.elan/toolchains/leanprover--lean4---v4.28.0/bin}"
COMPARATOR_ROOT="${COMPARATOR_ROOT:-/private/tmp/lean-comparator-428}"
FAKE_LANDRUN="${FAKE_LANDRUN:-/private/tmp/lean-comparator/scripts/fake-landrun.sh}"
COMPARATOR_BIN="${COMPARATOR_BIN:-$COMPARATOR_ROOT/.lake/build/bin/comparator}"
LEAN4EXPORT_BIN="${LEAN4EXPORT_BIN:-$COMPARATOR_ROOT/.lake/packages/lean4export/.lake/build/bin/lean4export}"
PATH_SHIM="${PATH_SHIM:-/private/tmp/comparator-428-path}"
ALLOW_FAKE="${COMPARATOR_ALLOW_FAKE_LANDRUN:-0}"

if [[ -n "${COMPARATOR_LANDRUN:-}" ]]; then
  LANDRUN_BIN="$COMPARATOR_LANDRUN"
elif command -v landrun >/dev/null 2>&1; then
  LANDRUN_BIN="$(command -v landrun)"
elif [[ "$ALLOW_FAKE" == "1" ]]; then
  LANDRUN_BIN="$FAKE_LANDRUN"
  echo "WARNING: using fake landrun; this is a local smoke test, not a hardened certificate." >&2
else
  cat >&2 <<'EOF'
ERROR: real landrun was not found.

Install/run Comparator in a Linux environment with real landrun available, or
set COMPARATOR_LANDRUN=/path/to/landrun.  For local development only, rerun with
COMPARATOR_ALLOW_FAKE_LANDRUN=1 to use Comparator's unsandboxed fake-landrun shim.
EOF
  exit 1
fi

if [[ ! -x "$LANDRUN_BIN" ]]; then
  echo "ERROR: landrun executable is not executable: $LANDRUN_BIN" >&2
  exit 1
fi
if [[ ! -x "$COMPARATOR_BIN" ]]; then
  echo "ERROR: comparator binary is not executable: $COMPARATOR_BIN" >&2
  exit 1
fi
if [[ ! -x "$LEAN4EXPORT_BIN" ]]; then
  echo "ERROR: lean4export binary is not executable: $LEAN4EXPORT_BIN" >&2
  exit 1
fi

mkdir -p "$PATH_SHIM"
ln -sf "$LANDRUN_BIN" "$PATH_SHIM/landrun"
ln -sf "$LEAN4EXPORT_BIN" "$PATH_SHIM/lean4export"

PATH="$PATH_SHIM:$LEAN428_BIN:$PATH" \
  "$LEAN428_BIN/lake" env "$COMPARATOR_BIN" audit/comparator-paper-config.template.json
