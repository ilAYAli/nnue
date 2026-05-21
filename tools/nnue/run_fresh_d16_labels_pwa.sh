#!/usr/bin/env bash
set -euo pipefail

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
ENGINE_REPO="${ENGINE_REPO:-$HOME/code/cpp/chess/enyo}"
TOOLS="$NNUE_REPO/tools/nnue"
RUN="${RUN:-$HOME/tmp/enyo_teacher/fresh_d16_labels_$(date +%Y%m%d_%H%M%S)}"

SELFPLAY_RUNNER="${SELFPLAY_RUNNER:-$NNUE_REPO/tools/posgen/run_selfplay.sh}"
PGN_TO_JSONL="${PGN_TO_JSONL:-$NNUE_REPO/tools/posgen/pgn_to_jsonl.py}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
NNUE_FILE="${NNUE_FILE:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
STOCKFISH="${STOCKFISH:-$HOME/local/bin/stockfish}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

SELFPLAY_GAMES="${SELFPLAY_GAMES:-160000}"
SELFPLAY_SHARD_GAMES="${SELFPLAY_SHARD_GAMES:-1000}"
SELFPLAY_CONCURRENCY="${SELFPLAY_CONCURRENCY:-12}"
SELFPLAY_THREADS="${SELFPLAY_THREADS:-1}"
SELFPLAY_HASH="${SELFPLAY_HASH:-128}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-8}"
SELFPLAY_SEED="${SELFPLAY_SEED:-2026051501}"

SOURCE_MAX_ABS_CP="${SOURCE_MAX_ABS_CP:-1600}"
LABEL_DEPTH="${LABEL_DEPTH:-16}"
LABEL_SHARDS="${LABEL_SHARDS:-24}"
LABEL_THREADS="${LABEL_THREADS:-1}"
LABEL_HASH="${LABEL_HASH:-128}"
LABEL_MAX_ABS_CP="${LABEL_MAX_ABS_CP:-1600}"
LABEL_PROGRESS="${LABEL_PROGRESS:-10000}"

mkdir -p "$RUN/shards"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

label_pids=()
cleanup() {
  local rc=$?
  if [ "${#label_pids[@]}" -gt 0 ]; then
    kill "${label_pids[@]}" >/dev/null 2>&1 || true
  fi
  notify "Enyo NNUE fresh d16 labels finished rc=$rc run=$RUN"
  exit "$rc"
}
trap cleanup EXIT

require_file() {
  if [ ! -e "$1" ]; then
    notify "Enyo NNUE fresh d16 labels missing $1"
    exit 1
  fi
}

for required in "$PY" "$SELFPLAY_RUNNER" "$PGN_TO_JSONL" "$ENGINE" "$NNUE_FILE" \
  "$STOCKFISH" "$BOOK" "$TOOLS/sample_signed_buckets.py" \
  "$TOOLS/label_with_uci.py" "$TOOLS/pack_dataset.py"; do
  require_file "$required"
done

notify "Enyo NNUE fresh d16 labels start run=$RUN games=$SELFPLAY_GAMES label_depth=$LABEL_DEPTH"

SELFPLAY_PGN="$RUN/selfplay.pgn"
SELFPLAY_JSONL="$RUN/selfplay.jsonl"
SIGNED_JSONL="$RUN/source_signed.jsonl"
LABELED_JSONL="$RUN/labeled.jsonl"
PACKED="$RUN/packed"

if [ ! -s "$SELFPLAY_PGN" ]; then
  notify "Enyo NNUE fresh d16 labels: self-play games=$SELFPLAY_GAMES concurrency=$SELFPLAY_CONCURRENCY threads=$SELFPLAY_THREADS depth=$SELFPLAY_DEPTH"
  "$SELFPLAY_RUNNER" \
    --runner "$HOME/local/bin/fastchess" \
    --engine "$ENGINE" \
    --nnue-file "$NNUE_FILE" \
    --output "$SELFPLAY_PGN" \
    --games "$SELFPLAY_GAMES" \
    --shard-games "$SELFPLAY_SHARD_GAMES" \
    --concurrency "$SELFPLAY_CONCURRENCY" \
    --threads "$SELFPLAY_THREADS" \
    --depth "$SELFPLAY_DEPTH" \
    --book "$BOOK" \
    --srand "$SELFPLAY_SEED" \
    --restart off \
    --engine-option "Hash=$SELFPLAY_HASH"
  notify "Enyo NNUE fresh d16 labels: self-play PGN ready $SELFPLAY_PGN"
fi

if [ ! -s "$SELFPLAY_JSONL" ]; then
  notify "Enyo NNUE fresh d16 labels: convert PGN to JSONL"
  "$PY" "$PGN_TO_JSONL" "$SELFPLAY_PGN" \
    --output "$SELFPLAY_JSONL" \
    --stats "$RUN/selfplay_jsonl_stats.json" \
    --skip-plies 8 \
    --min-depth "$SELFPLAY_DEPTH" \
    --max-abs-cp "$SOURCE_MAX_ABS_CP"
  wc -l "$SELFPLAY_JSONL" | tee "$RUN/selfplay_jsonl.wc"
  rows=$(wc -l < "$SELFPLAY_JSONL" | tr -d ' ')
  notify "Enyo NNUE fresh d16 labels: JSONL ready rows=$rows"
fi

if [ ! -s "$SIGNED_JSONL" ]; then
  notify "Enyo NNUE fresh d16 labels: signed-bucket sample"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$SELFPLAY_JSONL" \
    --output "$SIGNED_JSONL" \
    --score-field score \
    --unique-fen \
    --seed "$SELFPLAY_SEED" \
    --progress 1000000 \
    --bucket any0:any:0:25:400000 \
    --bucket p25:pos:25:75:250000 \
    --bucket n25:neg:25:75:250000 \
    --bucket p75:pos:75:150:300000 \
    --bucket n75:neg:75:150:300000 \
    --bucket p150:pos:150:300:350000 \
    --bucket n150:neg:150:300:350000 \
    --bucket p300:pos:300:600:250000 \
    --bucket n300:neg:300:600:250000 \
    --bucket p600:pos:600:1200:150000 \
    --bucket n600:neg:600:1200:150000 | tee "$RUN/sample_signed.log"
  wc -l "$SIGNED_JSONL" | tee "$RUN/source_signed.wc"
  rows=$(wc -l < "$SIGNED_JSONL" | tr -d ' ')
  notify "Enyo NNUE fresh d16 labels: signed-bucket sample ready rows=$rows"
fi

if [ ! -s "$LABELED_JSONL" ]; then
  notify "Enyo NNUE fresh d16 labels: Stockfish depth=$LABEL_DEPTH shards=$LABEL_SHARDS"
  for shard in $(seq 0 $((LABEL_SHARDS - 1))); do
    "$PY" "$TOOLS/label_with_uci.py" \
      --input "$SIGNED_JSONL" \
      --output "$RUN/shards/label.${shard}.jsonl" \
      --engine "$STOCKFISH" \
      --depth "$LABEL_DEPTH" \
      --threads "$LABEL_THREADS" \
      --hash "$LABEL_HASH" \
      --shard-count "$LABEL_SHARDS" \
      --shard-index "$shard" \
      --max-abs-cp "$LABEL_MAX_ABS_CP" \
      --progress "$LABEL_PROGRESS" > "$RUN/shards/label.${shard}.log" 2>&1 &
    label_pids+=("$!")
  done

  total=$(wc -l < "$SIGNED_JSONL" | tr -d ' ')
  last_pct=-1
  while true; do
    written=$(find "$RUN/shards" -name 'label.*.jsonl' -type f -print0 \
      | xargs -0 cat 2>/dev/null | wc -l | tr -d ' ')
    pct=$((written * 100 / total))
    if [ "$pct" -ge $((last_pct + 10)) ] || [ "$pct" -ge 99 ]; then
      notify "Enyo NNUE fresh d16 labels: label written=$written/$total (${pct}%)"
      last_pct="$pct"
    fi

    alive=0
    for pid in "${label_pids[@]}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done
    if [ "$alive" -eq 0 ]; then
      break
    fi
    sleep 60
  done

  for pid in "${label_pids[@]}"; do
    wait "$pid"
  done
  label_pids=()
  cat "$RUN"/shards/label.*.jsonl > "$LABELED_JSONL"
  wc -l "$LABELED_JSONL" | tee "$RUN/labeled.wc"
  rows=$(wc -l < "$LABELED_JSONL" | tr -d ' ')
  notify "Enyo NNUE fresh d16 labels: labels ready rows=$rows"
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE fresh d16 labels: pack labeled rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$LABELED_JSONL" \
    --out-dir "$PACKED" \
    --progress 200000 | tee "$RUN/pack.log"
  notify "Enyo NNUE fresh d16 labels: packed labels ready $PACKED"
fi

notify "Enyo NNUE fresh d16 labels ready run=$RUN packed=$PACKED"
echo "DONE fresh_d16_labels $RUN"
