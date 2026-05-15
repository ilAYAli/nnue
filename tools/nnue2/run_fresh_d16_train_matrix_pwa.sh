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
ENGINE_REPO="${ENGINE_REPO:-$HOME/code/cpp/chess/enyo}"
TOOLS="$NNUE_REPO/tools/nnue2"
SOURCE_RUN="${SOURCE_RUN:-$(ls -td "$HOME"/tmp/enyo_teacher/fresh_d16_labels_* 2>/dev/null | head -1)}"
LICHESS_RUN="${LICHESS_RUN:-$(ls -td "$HOME"/tmp/enyo_teacher/lichess_eval_slice_* 2>/dev/null | head -1)}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/fresh_d16_train_matrix_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-2}"
VAL_ROWS="${VAL_ROWS:-100000}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE fresh d16 train matrix finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE fresh d16 train matrix start run=$RUN"

SOURCE_JSON="$SOURCE_RUN/labeled.jsonl"
SOURCE_PACKED="$SOURCE_RUN/packed"
LICHESS_JSON="$LICHESS_RUN/lichess_eval_signed.jsonl"
LICHESS_PACKED="$LICHESS_RUN/packed"

for required in "$PY" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" \
  "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" "$SOURCE_JSON" \
  "$SOURCE_PACKED/meta.json" "$LICHESS_JSON" "$LICHESS_PACKED/meta.json" \
  "$INIT"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE fresh d16 train matrix missing $required"
    exit 1
  fi
done

rows_in_meta() {
  "$PY" - "$1/meta.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(int(json.load(handle)["rows"]))
PY
}

eval_skip() {
  local rows="$1"
  local eval_rows="${2:-50000}"
  if [ "$rows" -gt "$eval_rows" ]; then
    echo $((rows - eval_rows))
  else
    echo 0
  fi
}

train_one() {
  local tag="$1"
  local data="$2"
  local objective="$3"
  local clamp="$4"
  local lr="$5"
  local epochs="$6"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ -s "$dir/model.nn" ]; then
    return
  fi

  notify "Enyo NNUE fresh d16 train matrix: train $tag"
  local args=(
    "$TOOLS/train.py"
    --data "$data"
    --init-from-nn "$INIT"
    --objective "$objective"
    --select-metric mae
    --epochs "$epochs"
    --patience 2
    --batch-size "$BATCH_SIZE"
    --lr "$lr"
    --weight-decay 1e-6
    --target-clamp "$clamp"
    --device "$DEVICE"
    --workers "$WORKERS"
    --val-rows "$VAL_ROWS"
    --trainable all
    --out "$dir/model.pt"
    --out-nn "$dir/model.nn"
  )
  if [ "$objective" = "huber" ]; then
    args+=(--huber-beta 200)
  else
    args+=(--wdl-lambda 0.95)
    if grep -q '"lichess_eval"' "$data/source_map.json" 2>/dev/null; then
      args+=(--source-wdl-lambda lichess_eval=1.0)
    fi
  fi
  "$PY" "${args[@]}" | tee "$dir/train.log"
}

eval_one() {
  local net="$1"
  local data="$2"
  local skip="$3"
  local out="$4"
  "$PY" "$TOOLS/eval_dataset.py" \
    --net "$net" \
    --data "$data" \
    --skip "$skip" \
    --rows 50000 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --target-clamp 1600 \
    --buckets \
    --sources > "$out"
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  mkdir -p "$dir/eval"

  local fresh_rows lichess_rows fresh_skip lichess_skip
  fresh_rows=$(rows_in_meta "$SOURCE_PACKED")
  lichess_rows=$(rows_in_meta "$LICHESS_PACKED")
  fresh_skip=$(eval_skip "$fresh_rows" 50000)
  lichess_skip=$(eval_skip "$lichess_rows" 50000)

  notify "Enyo NNUE fresh d16 train matrix: eval $tag"
  eval_one "$INIT" "$SOURCE_PACKED" "$fresh_skip" "$dir/eval/fresh_baseline.txt"
  eval_one "$dir/model.nn" "$SOURCE_PACKED" "$fresh_skip" "$dir/eval/fresh_candidate.txt"
  eval_one "$INIT" "$LICHESS_PACKED" "$lichess_skip" "$dir/eval/lichess_baseline.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" "$lichess_skip" "$dir/eval/lichess_candidate.txt"
}

make_mix() {
  local fresh_rows
  local lichess_rows
  fresh_rows=$(rows_in_meta "$SOURCE_PACKED")
  if [ "$fresh_rows" -gt 2550000 ]; then
    fresh_rows=2550000
  fi
  lichess_rows=$((fresh_rows * 15 / 85))
  if [ "$lichess_rows" -gt 450000 ]; then
    lichess_rows=450000
  fi

  MIX_JSON="$RUN/mix_fresh${fresh_rows}_lichess${lichess_rows}.jsonl"
  MIX_PACKED="$RUN/mix_fresh_lichess_packed"
  if [ ! -s "$MIX_JSON" ]; then
    notify "Enyo NNUE fresh d16 train matrix: create 85/15 fresh+lichess mix"
    "$PY" "$TOOLS/mix_jsonl.py" \
      --output "$MIX_JSON" \
      --source "$SOURCE_JSON:$fresh_rows" \
      --source "$LICHESS_JSON:$lichess_rows" \
      --seed 2026051502 \
      --progress 500000 | tee "$RUN/mix.log"
  fi
  if [ ! -s "$MIX_PACKED/meta.json" ]; then
    notify "Enyo NNUE fresh d16 train matrix: pack 85/15 mix"
    "$PY" "$TOOLS/pack_dataset.py" \
      --input "$MIX_JSON" \
      --out-dir "$MIX_PACKED" \
      --progress 500000 | tee "$RUN/mix_pack.log"
  fi
}

{
  echo "source_run=$SOURCE_RUN"
  echo "lichess_run=$LICHESS_RUN"
  echo "init=$INIT"
} > "$RUN/inputs.txt"

train_one fresh_huber_cp800_lr7e7_e8 "$SOURCE_PACKED" huber 800 7e-7 8
eval_candidate fresh_huber_cp800_lr7e7_e8

train_one fresh_huber_cp1000_lr1e6_e6 "$SOURCE_PACKED" huber 1000 1e-6 6
eval_candidate fresh_huber_cp1000_lr1e6_e6

train_one fresh_mpe25_cp1200_lr7e7_e6 "$SOURCE_PACKED" mpe25 1200 7e-7 6
eval_candidate fresh_mpe25_cp1200_lr7e7_e6

make_mix
train_one fresh85_lichess15_mpe25_cp1200_lr7e7_e6 "$MIX_PACKED" mpe25 1200 7e-7 6
eval_candidate fresh85_lichess15_mpe25_cp1200_lr7e7_e6

notify "Enyo NNUE fresh d16 train matrix ready run=$RUN"
