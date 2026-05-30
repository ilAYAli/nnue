#!/usr/bin/env bash
set -euo pipefail

log_event() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
}

net_size() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.getsize(sys.argv[1]))
PY
}

net_layout() {
  case "$(net_size "$1")" in
    25203012) echo "16-bucket" ;;
    50368836) echo "32-bucket" ;;
    *) echo "unknown" ;;
  esac
}

check_engine_loads_net() {
  local label="$1"
  local engine="$2"
  local net="$3"
  local layout
  local output
  local rc=0

  layout=$(net_layout "$net")
  if [ "$layout" = "unknown" ]; then
    log_event "Enyo NNUE SPRT invalid $label net size $(net_size "$net"): $net"
    exit 1
  fi

  if command -v timeout >/dev/null 2>&1; then
    output=$(printf 'uci\nevalnet load %s\nquit\n' "$net" | timeout 20 "$engine" 2>&1) || rc=$?
  elif command -v gtimeout >/dev/null 2>&1; then
    output=$(printf 'uci\nevalnet load %s\nquit\n' "$net" | gtimeout 20 "$engine" 2>&1) || rc=$?
  else
    output=$(printf 'uci\nevalnet load %s\nquit\n' "$net" | "$engine" 2>&1) || rc=$?
  fi

  if [ "$rc" -ne 0 ] || ! grep -q "evalnet load ok" <<<"$output"; then
    log_event "Enyo NNUE SPRT engine cannot load $label $layout net: engine=$engine net=$net"
    printf '%s\n' "$output" | tail -40
    exit 1
  fi
  log_event "Enyo NNUE SPRT load ok $label $layout net=$net"
}

NET="${NET:?set NET to the candidate .nn file}"
TAG="${TAG:-$(basename "$(dirname "$NET")")}"
RUN="${RUN:-$(dirname "$(dirname "$NET")")}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
INIT="${INIT:-$HOME/code/cpp/chess/enyo/net/berserk-d43206fe90e4.nn}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"
GAMES="${GAMES:-4000}"
CONCURRENCY="${CONCURRENCY:-10}"
THREADS="${THREADS:-2}"
HASH="${HASH:-512}"
TC="${TC:-2+0.02}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-8}"
RESTART="${RESTART:-off}"
LOG_DIR="${LOG_DIR:-$RUN/$TAG/sprt_confirm_$(date +%Y%m%d_%H%M%S)}"
NTFY_URL="${NTFY_URL:-https://ntfy.wahlman.no/sprt}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

for required in "$NET" "$ENGINE" "$INIT" "$SPRT"; do
  if [ ! -e "$required" ]; then
    log_event "Enyo NNUE SPRT missing $required"
    exit 1
  fi
done
if [ -n "$BOOK" ] && [ ! -e "$BOOK" ]; then
  log_event "Enyo NNUE SPRT missing $BOOK"
  exit 1
fi

check_engine_loads_net "candidate" "$ENGINE" "$NET"
check_engine_loads_net "reference" "$ENGINE" "$INIT"

log_event "Enyo NNUE SPRT start $TAG games=$GAMES net=$NET"

args=(
  --candidate "$ENGINE"
  --reference "$ENGINE"
  --candidate-uci "nnue_file=$NET"
  --reference-uci "nnue_file=$INIT"
  --games "$GAMES"
  --elo0 "$ELO0"
  --elo1 "$ELO1"
  --concurrency "$CONCURRENCY"
  --threads "$THREADS"
  --tc "$TC"
  --restart "$RESTART"
  --log-dir "$LOG_DIR"
  --name "${TAG}_vs_refnet"
  --ntfy-url "$NTFY_URL"
)
if [ -n "$HASH" ]; then
  args+=(--candidate-uci "Hash=$HASH")
  args+=(--reference-uci "Hash=$HASH")
fi
if [ -n "$BOOK" ]; then
  args+=(--book "$BOOK")
fi

set +e
"$SPRT" "${args[@]}" 2>&1 | tee "$LOG_DIR/sprt.log"
rc=${PIPESTATUS[0]}
set -e

final=$(rg --no-config \
  "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" \
  "$LOG_DIR/sprt.log" 2>/dev/null | tail -6 | tr '\n' ' ' || true)
log_event "Enyo NNUE SPRT finished $TAG rc=$rc ${final:-no final status}"
exit "$rc"
