#!/usr/bin/env bash
set -euo pipefail

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
}

NET="${NET:?set NET to the candidate .nn file}"
TAG="${TAG:-$(basename "$(dirname "$NET")")}"
RUN="${RUN:-$(dirname "$(dirname "$NET")")}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
INIT="${INIT:-$HOME/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn}"
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

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

for required in "$NET" "$ENGINE" "$INIT" "$SPRT" "$BOOK"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE SPRT missing $required"
    exit 1
  fi
done

notify "Enyo NNUE SPRT start $TAG games=$GAMES net=$NET"

set +e
"$SPRT" \
  --candidate "$ENGINE" \
  --reference "$ENGINE" \
  --candidate-option "Hash=$HASH" \
  --candidate-option "nnue2_file=$NET" \
  --reference-option "Hash=$HASH" \
  --reference-option "nnue2_file=$INIT" \
  --book "$BOOK" \
  --games "$GAMES" \
  --elo0 "$ELO0" \
  --elo1 "$ELO1" \
  --concurrency "$CONCURRENCY" \
  --threads "$THREADS" \
  --tc "$TC" \
  --restart "$RESTART" \
  --log-dir "$LOG_DIR" \
  --name "${TAG}_vs_refnet" \
  --eta 2>&1 | tee "$LOG_DIR/sprt.log"
rc=${PIPESTATUS[0]}
set -e

final=$(rg --no-config \
  "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" \
  "$LOG_DIR/sprt.log" 2>/dev/null | tail -6 | tr '\n' ' ' || true)
notify "Enyo NNUE SPRT finished $TAG rc=$rc ${final:-no final status}"
exit "$rc"
