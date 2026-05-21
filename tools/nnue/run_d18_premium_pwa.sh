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
SOURCE_RUN="${SOURCE_RUN:-$(ls -td "$HOME"/tmp/enyo_teacher/fresh_d16_labels_* 2>/dev/null | head -1)}"
SOURCE_JSON="${SOURCE_JSON:-$SOURCE_RUN/selfplay.jsonl}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/d18_premium_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
STOCKFISH="${STOCKFISH:-$HOME/local/bin/stockfish}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
WORKERS="${WORKERS:-2}"
LABEL_SHARDS="${LABEL_SHARDS:-24}"
LABEL_THREADS="${LABEL_THREADS:-1}"
LABEL_HASH="${LABEL_HASH:-256}"
LABEL_DEPTH="${LABEL_DEPTH:-18}"

mkdir -p "$RUN/shards"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

label_pids=()
cleanup() {
  local rc=$?
  if [ "${#label_pids[@]}" -gt 0 ]; then
    kill "${label_pids[@]}" >/dev/null 2>&1 || true
  fi
  notify "Enyo NNUE d18 premium finished rc=$rc run=$RUN"
  exit "$rc"
}
trap cleanup EXIT

for required in "$PY" "$SOURCE_JSON" "$INIT" "$STOCKFISH" \
  "$TOOLS/sample_signed_buckets.py" "$TOOLS/label_with_uci.py" \
  "$TOOLS/pack_dataset.py" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" \
  "$TOOLS/run_net_sprt_pwa.sh"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE d18 premium missing $required"
    exit 1
  fi
done

SIGNED_JSONL="$RUN/source_signed.jsonl"
LABELED_JSONL="$RUN/labeled.jsonl"
PACKED="$RUN/packed"

notify "Enyo NNUE d18 premium start run=$RUN source=$SOURCE_JSON"

if [ ! -s "$SIGNED_JSONL" ]; then
  notify "Enyo NNUE d18 premium: signed-bucket sample about 1M rows"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$SOURCE_JSON" \
    --output "$SIGNED_JSONL" \
    --score-field score \
    --unique-fen \
    --seed 2026051601 \
    --progress 1000000 \
    --bucket any0:any:0:25:120000 \
    --bucket p25:pos:25:75:80000 \
    --bucket n25:neg:25:75:80000 \
    --bucket p75:pos:75:150:110000 \
    --bucket n75:neg:75:150:110000 \
    --bucket p150:pos:150:300:120000 \
    --bucket n150:neg:150:300:120000 \
    --bucket p300:pos:300:600:90000 \
    --bucket n300:neg:300:600:90000 \
    --bucket p600:pos:600:1200:50000 \
    --bucket n600:neg:600:1200:50000 | tee "$RUN/sample_signed.log"
  wc -l "$SIGNED_JSONL" | tee "$RUN/source_signed.wc"
fi

if [ ! -s "$LABELED_JSONL" ]; then
  notify "Enyo NNUE d18 premium: Stockfish depth=$LABEL_DEPTH shards=$LABEL_SHARDS"
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
      --max-abs-cp 1600 \
      --progress 10000 > "$RUN/shards/label.${shard}.log" 2>&1 &
    label_pids+=("$!")
  done

  total=$(wc -l < "$SIGNED_JSONL" | tr -d ' ')
  last_pct=-1
  while true; do
    written=$(find "$RUN/shards" -name 'label.*.jsonl' -type f -print0 \
      | xargs -0 cat 2>/dev/null | wc -l | tr -d ' ')
    pct=$((written * 100 / total))
    if [ "$pct" -ge $((last_pct + 10)) ] || [ "$pct" -ge 99 ]; then
      notify "Enyo NNUE d18 premium: label written=$written/$total (${pct}%)"
      last_pct="$pct"
    fi
    alive=0
    for pid in "${label_pids[@]}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done
    [ "$alive" -eq 0 ] && break
    sleep 60
  done

  for pid in "${label_pids[@]}"; do
    wait "$pid"
  done
  label_pids=()
  cat "$RUN"/shards/label.*.jsonl > "$LABELED_JSONL"
  wc -l "$LABELED_JSONL" | tee "$RUN/labeled.wc"
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE d18 premium: pack labels"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$LABELED_JSONL" \
    --out-dir "$PACKED" \
    --progress 200000 | tee "$RUN/pack.log"
fi

train_one() {
  local tag="$1"
  local clamp="$2"
  local lr="$3"
  local epochs="$4"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE d18 premium: train $tag"
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
      --target-clamp "$clamp" \
      --device "$DEVICE" \
      --workers "$WORKERS" \
      --val-rows 100000 \
      --trainable all \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" | tee "$dir/train.log"
  fi
}

rows_in_meta() {
  "$PY" - "$1/meta.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(int(json.load(handle)["rows"]))
PY
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  local rows skip
  rows=$(rows_in_meta "$PACKED")
  skip=$((rows > 100000 ? rows - 100000 : 0))
  notify "Enyo NNUE d18 premium: eval $tag"
  "$PY" "$TOOLS/eval_dataset.py" \
    --net "$INIT" \
    --data "$PACKED" \
    --skip "$skip" \
    --rows 100000 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --target-clamp 1600 \
    --buckets > "$dir/eval_baseline.txt"
  "$PY" "$TOOLS/eval_dataset.py" \
    --net "$dir/model.nn" \
    --data "$PACKED" \
    --skip "$skip" \
    --rows 100000 \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --target-clamp 1600 \
    --buckets > "$dir/eval_candidate.txt"
}

train_one huber_cp1000_lr1e6_e8 1000 1e-6 8
eval_candidate huber_cp1000_lr1e6_e8

train_one huber_cp800_lr7e7_e8 800 7e-7 8
eval_candidate huber_cp800_lr7e7_e8

best_tag="$("$PY" - "$RUN" <<'PY'
import pathlib
import re
import sys
run = pathlib.Path(sys.argv[1])
best = None
for path in run.glob("huber_*/eval_candidate.txt"):
    text = path.read_text(errors="replace")
    match = re.search(r"^mae=([0-9.]+)", text, re.M)
    if not match:
        continue
    item = (float(match.group(1)), path.parent.name)
    best = item if best is None or item < best else best
print(best[1] if best else "huber_cp1000_lr1e6_e8")
PY
)"

best_net="$RUN/$best_tag/model.nn"
notify "Enyo NNUE d18 premium: start 1000-game smoke SPRT best=$best_tag"
NET="$best_net" \
TAG="d18_premium_${best_tag}_smoke" \
GAMES="${SPRT_GAMES:-1000}" \
CONCURRENCY="${SPRT_CONCURRENCY:-10}" \
THREADS="${SPRT_THREADS:-2}" \
HASH="${SPRT_HASH:-512}" \
TC="${SPRT_TC:-2+0.02}" \
ELO1="${SPRT_ELO1:-5}" \
"$TOOLS/run_net_sprt_pwa.sh"

notify "Enyo NNUE d18 premium ready run=$RUN best=$best_tag"
