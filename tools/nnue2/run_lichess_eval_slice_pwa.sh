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

run_low() {
  if command -v ionice >/dev/null 2>&1; then
    nice -n "${NICE_LEVEL:-10}" ionice -c2 -n7 "$@"
  else
    nice -n "${NICE_LEVEL:-10}" "$@"
  fi
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
TOOLS="$NNUE_REPO/tools/nnue2"
RUN="${RUN:-$HOME/tmp/enyo_teacher/lichess_eval_slice_$(date +%Y%m%d_%H%M%S)}"
LICHESS_EVAL="${LICHESS_EVAL:-$HOME/code/cpp/chess/assets/lichess_db_eval.jsonl.zst}"
MIN_DEPTH="${MIN_DEPTH:-18}"
MAX_ABS_CP="${MAX_ABS_CP:-1600}"
SEED="${SEED:-20260515}"
STOP_WHEN_FULL="${STOP_WHEN_FULL:-0}"
MAX_INPUT_ROWS="${MAX_INPUT_ROWS:-0}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE lichess eval slice finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE lichess eval slice start run=$RUN"

for required in "$PY" "$TOOLS" "$LICHESS_EVAL"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE lichess eval slice missing $required"
    exit 1
  fi
done

ROWS="$RUN/lichess_eval_signed.jsonl"
PACKED="$RUN/packed"
IMPORT_ARGS=(
  "$TOOLS/import_lichess_eval.py"
  --input "$LICHESS_EVAL"
  --output "$ROWS"
  --min-depth "$MIN_DEPTH"
  --max-abs-cp "$MAX_ABS_CP"
  --seed "$SEED"
  --progress 1000000
  --bucket any0:any:0:25:150000
  --bucket pos25_75:pos:25:75:100000
  --bucket neg25_75:neg:25:75:100000
  --bucket pos75_150:pos:75:150:120000
  --bucket neg75_150:neg:75:150:120000
  --bucket pos150_300:pos:150:300:120000
  --bucket neg150_300:neg:150:300:120000
  --bucket pos300_600:pos:300:600:80000
  --bucket neg300_600:neg:300:600:80000
  --bucket pos600_1200:pos:600:1200:40000
  --bucket neg600_1200:neg:600:1200:40000
)
if [ "$STOP_WHEN_FULL" = "1" ]; then
  IMPORT_ARGS+=(--stop-when-full)
fi
if [ "$MAX_INPUT_ROWS" != "0" ]; then
  IMPORT_ARGS+=(--max-input-rows "$MAX_INPUT_ROWS")
fi

if [ ! -s "$ROWS" ]; then
  notify "Enyo NNUE lichess eval slice: stream signed buckets"
  run_low "$PY" "${IMPORT_ARGS[@]}" | tee "$RUN/import.log"
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE lichess eval slice: pack rows"
  run_low "$PY" "$TOOLS/pack_dataset.py" \
    --input "$ROWS" \
    --out-dir "$PACKED" \
    --progress 250000 | tee "$RUN/pack.log"
fi

{
  echo "run=$RUN"
  echo "rows=$ROWS"
  echo "packed=$PACKED"
  [ -f "$PACKED/meta.json" ] && cat "$PACKED/meta.json"
} > "$RUN/done.txt"

notify "Enyo NNUE lichess eval slice ready run=$RUN"
