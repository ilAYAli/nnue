#!/usr/bin/env bash
set -euo pipefail

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
TOOLS="$NNUE_REPO/tools/nnue"
ENGINE_REPO="${ENGINE_REPO:-$HOME/code/cpp/chess/enyo}"
BASE_RUN="${BASE_RUN:-$HOME/tmp/enyo_teacher/fresh_d16_resample4m_20260516_224729}"
HARD_JSON="${HARD_JSON:-$HOME/tmp/nnue_arch/instability_d6d12_100k.jsonl}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/fresh_d16_hardblend_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
STOCKFISH="${STOCKFISH:-$HOME/local/bin/stockfish}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-2}"
HARD_ROWS="${HARD_ROWS:-100000}"
BASE_ROWS="${BASE_ROWS:-3900000}"
LABEL_SHARDS="${LABEL_SHARDS:-12}"
SPRT_GAMES="${SPRT_GAMES:-1000}"

mkdir -p "$RUN/shards"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

trap 'rc=$?; notify "Enyo NNUE hardblend finished rc=$rc run=$RUN"; exit $rc' EXIT

for required in "$PY" "$BASE_RUN/labeled.jsonl" "$HARD_JSON" "$INIT" "$STOCKFISH" \
  "$TOOLS/label_with_uci.py" "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" \
  "$TOOLS/train.py" "$TOOLS/eval_dataset.py" "$TOOLS/run_net_sprt_pwa.sh"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE hardblend missing $required"
    exit 1
  fi
done

notify "Enyo NNUE hardblend start run=$RUN base=$BASE_RUN hard=$HARD_JSON"

HARD_D16="$RUN/hard_d16.jsonl"
MIX_JSON="$RUN/mix.jsonl"
PACKED="$RUN/packed"

if [ ! -s "$HARD_D16" ]; then
  notify "Enyo NNUE hardblend: relabel hard rows Stockfish depth=16"
  pids=()
  for shard in $(seq 0 $((LABEL_SHARDS - 1))); do
    "$PY" "$TOOLS/label_with_uci.py" \
      --input "$HARD_JSON" \
      --output "$RUN/shards/hard_d16.${shard}.jsonl" \
      --engine "$STOCKFISH" \
      --depth 16 \
      --threads 1 \
      --hash 128 \
      --shard-count "$LABEL_SHARDS" \
      --shard-index "$shard" \
      --max-abs-cp 1600 \
      --progress 5000 > "$RUN/shards/hard_d16.${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  cat "$RUN"/shards/hard_d16.*.jsonl > "$HARD_D16"
  wc -l "$HARD_D16" | tee "$RUN/hard_d16.wc"
fi

if [ ! -s "$MIX_JSON" ]; then
  notify "Enyo NNUE hardblend: mix base=$BASE_ROWS hard=$HARD_ROWS"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIX_JSON" \
    --seed 2026051701 \
    --progress 250000 \
    --source "$BASE_RUN/labeled.jsonl:$BASE_ROWS" \
    --source "$HARD_D16:$HARD_ROWS" | tee "$RUN/mix.log"
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE hardblend: pack mixed rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$MIX_JSON" \
    --out-dir "$PACKED" \
    --progress 250000 | tee "$RUN/pack.log"
fi

train_one() {
  local tag="$1" lr="$2" epochs="$3"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  notify "Enyo NNUE hardblend: train $tag"
  "$PY" "$TOOLS/train.py" \
    --data "$PACKED" \
    --init-from-nn "$INIT" \
    --objective huber \
    --huber-beta 200 \
    --select-metric mae \
    --epochs "$epochs" \
    --patience 2 \
    --batch-size "$BATCH_SIZE" \
    --lr "$lr" \
    --weight-decay 1e-6 \
    --target-clamp 1000 \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --val-rows 100000 \
    --trainable all \
    --out "$dir/model.pt" \
    --out-nn "$dir/model.nn" | tee "$dir/train.log"

  notify "Enyo NNUE hardblend: eval $tag"
  "$PY" "$TOOLS/eval_dataset.py" --net "$INIT" --data "$PACKED" \
    --skip 3900000 --rows 100000 --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" --target-clamp 1600 --buckets > "$dir/eval_baseline.txt"
  "$PY" "$TOOLS/eval_dataset.py" --net "$dir/model.nn" --data "$PACKED" \
    --skip 3900000 --rows 100000 --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" --target-clamp 1600 --buckets > "$dir/eval_candidate.txt"

  notify "Enyo NNUE hardblend: smoke SPRT $tag"
  RUN="$RUN" NET="$dir/model.nn" TAG="fresh_d16_hardblend_${tag}_smoke" \
    GAMES="$SPRT_GAMES" CONCURRENCY=10 THREADS=2 HASH=512 TC=2+0.02 ELO1=5 \
    "$TOOLS/run_net_sprt_pwa.sh"
}

train_one hardblend_huber_cp1000_lr5e7_e6 5e-7 6

notify "Enyo NNUE hardblend ready run=$RUN"
echo "DONE hardblend $RUN"
