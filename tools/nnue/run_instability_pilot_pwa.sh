#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
  if [ -n "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" ]; then
    curl -fsS -m 10 -u "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" \
      -d "$msg" https://ntfy.wahlman.no/ping >/dev/null 2>&1 || true
  else
    curl -fsS -m 10 -d "$msg" https://ntfy.wahlman.no/ping \
      >/dev/null 2>&1 || true
  fi
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
TOOLS="$NNUE_REPO/tools/nnue"
SOURCE="${SOURCE:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl}"
STOCKFISH="${STOCKFISH:-$HOME/local/bin/stockfish}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/instability_pilot_$(date +%Y%m%d_%H%M%S)}"
SHARDS="${SHARDS:-2}"
SELECTED_PER_SHARD="${SELECTED_PER_SHARD:-10000}"
MAX_OUTPUT_PER_SHARD="${MAX_OUTPUT_PER_SHARD:-5000}"
LOW_DEPTH="${LOW_DEPTH:-8}"
HIGH_DEPTH="${HIGH_DEPTH:-16}"
MIN_DELTA_CP="${MIN_DELTA_CP:-80}"
POSITION_TIMEOUT_S="${POSITION_TIMEOUT_S:-45}"

mkdir -p "$RUN/shards"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

pids=()
cleanup() {
  local rc=$?
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" >/dev/null 2>&1 || true
  fi
  notify "Enyo NNUE instability pilot finished rc=$rc run=$RUN"
  exit "$rc"
}
trap cleanup EXIT

for required in "$PY" "$TOOLS/sample_search_instability.py" "$SOURCE" "$STOCKFISH"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE instability pilot missing $required"
    exit 1
  fi
done

total_selected=$((SHARDS * SELECTED_PER_SHARD))
notify "Enyo NNUE instability pilot start run=$RUN source=$SOURCE selected=$total_selected workers=$SHARDS"

for shard in $(seq 0 $((SHARDS - 1))); do
  nice -n 10 "$PY" "$TOOLS/sample_search_instability.py" \
    --input "$SOURCE" \
    --output "$RUN/shards/instability.${shard}.jsonl" \
    --engine "$STOCKFISH" \
    --low-depth "$LOW_DEPTH" \
    --high-depth "$HIGH_DEPTH" \
    --threads 1 \
    --hash 128 \
    --shard-count "$SHARDS" \
    --shard-index "$shard" \
    --max-abs-cp 1600 \
    --min-delta-cp "$MIN_DELTA_CP" \
    --bestmove-change \
    --max-output "$MAX_OUTPUT_PER_SHARD" \
    --seed $((2026051900 + shard)) \
    --limit "$SELECTED_PER_SHARD" \
    --progress 2000 \
    --position-timeout-s "$POSITION_TIMEOUT_S" \
    > "$RUN/shards/instability.${shard}.log" 2>&1 &
  pids+=("$!")
done

notify "Enyo NNUE instability pilot workers launched pids=${pids[*]}"

for pid in "${pids[@]}"; do
  wait "$pid"
done
pids=()

cat "$RUN"/shards/instability.*.jsonl > "$RUN/instability.jsonl"
rows=$(wc -l < "$RUN/instability.jsonl" | tr -d " ")
notify "Enyo NNUE instability pilot rows ready rows=$rows run=$RUN"
printf "DONE instability_pilot %s rows=%s\n" "$RUN" "$rows"
