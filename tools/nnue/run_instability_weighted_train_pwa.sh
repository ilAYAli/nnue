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
TOOLS="$NNUE_REPO/tools/nnue"
BASE_JSON="${BASE_JSON:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl}"
HARD_JSON="${HARD_JSON:-$HOME/tmp/enyo_teacher/instability_pilot_20260519_153044/instability.jsonl}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/instability_weighted_train_$(date +%Y%m%d_%H%M%S)}"
BASE_ROWS="${BASE_ROWS:-1950000}"
HARD_ROWS="${HARD_ROWS:-50000}"
HARD_WEIGHT="${HARD_WEIGHT:-5.0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-2}"
VAL_ROWS="${VAL_ROWS:-100000}"
TAG="${TAG:-old1950k_instability50k_w5_huber_cp1000_lr5e7_e6}"

mkdir -p "$RUN/$TAG"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE instability weighted train finished rc=$rc run=$RUN"; exit $rc' EXIT

for required in "$PY" "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" \
  "$TOOLS/tag_jsonl_source.py" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" \
  "$BASE_JSON" "$HARD_JSON" "$INIT"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE instability weighted train missing $required"
    exit 1
  fi
done

notify "Enyo NNUE instability weighted train start run=$RUN base=$BASE_ROWS hard=$HARD_ROWS hard_weight=$HARD_WEIGHT tag=$TAG"

HARD_TAGGED="$RUN/instability_source.jsonl"
MIX_JSON="$RUN/source.jsonl"
PACKED="$RUN/packed"
DIR="$RUN/$TAG"

if [ ! -s "$HARD_TAGGED" ]; then
  notify "Enyo NNUE instability weighted train: tag instability source"
  "$PY" "$TOOLS/tag_jsonl_source.py" \
    --input "$HARD_JSON" \
    --output "$HARD_TAGGED" \
    --source-type instability \
    --progress 10000 | tee "$RUN/tag.log"
fi

if [ ! -s "$MIX_JSON" ]; then
  notify "Enyo NNUE instability weighted train: mix source rows"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIX_JSON" \
    --source "$BASE_JSON:$BASE_ROWS" \
    --source "$HARD_TAGGED:$HARD_ROWS" \
    --seed 2026051904 \
    --progress 250000 | tee "$RUN/mix.log"
  rows=$(wc -l < "$MIX_JSON" | tr -d " ")
  notify "Enyo NNUE instability weighted train: mix ready rows=$rows"
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE instability weighted train: pack source rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$MIX_JSON" \
    --out-dir "$PACKED" \
    --progress 250000 | tee "$RUN/pack.log"
  notify "Enyo NNUE instability weighted train: packed source ready $PACKED"
fi

if [ ! -s "$DIR/model.nn" ]; then
  notify "Enyo NNUE instability weighted train: train $TAG"
  "$PY" "$TOOLS/train.py" \
    --data "$PACKED" \
    --init-from-nn "$INIT" \
    --objective huber \
    --huber-beta 200 \
    --select-metric mae \
    --epochs 6 \
    --patience 2 \
    --batch-size "$BATCH_SIZE" \
    --lr 5e-7 \
    --weight-decay 1e-6 \
    --target-clamp 1000 \
    --source-loss-weight "instability=$HARD_WEIGHT" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --val-rows "$VAL_ROWS" \
    --trainable all \
    --out "$DIR/model.pt" \
    --out-nn "$DIR/model.nn" | tee "$DIR/train.log"
  notify "Enyo NNUE instability weighted train: train done $TAG"
fi

notify "Enyo NNUE instability weighted train: static eval $TAG"
"$PY" "$TOOLS/eval_dataset.py" \
  --net "$INIT" \
  --data "$PACKED" \
  --skip $((BASE_ROWS + HARD_ROWS - VAL_ROWS)) \
  --rows "$VAL_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --target-clamp 1600 \
  --buckets > "$DIR/baseline_eval.txt"
"$PY" "$TOOLS/eval_dataset.py" \
  --net "$DIR/model.nn" \
  --data "$PACKED" \
  --skip $((BASE_ROWS + HARD_ROWS - VAL_ROWS)) \
  --rows "$VAL_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --target-clamp 1600 \
  --buckets > "$DIR/candidate_eval.txt"

notify "Enyo NNUE instability weighted train ready run=$RUN net=$DIR/model.nn"
printf "DONE instability_weighted_train %s net=%s\n" "$RUN" "$DIR/model.nn"
