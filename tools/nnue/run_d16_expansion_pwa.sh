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
RUN="${RUN:-$HOME/tmp/enyo_teacher/d16_expansion_$(date +%Y%m%d_%H%M%S)}"
SOURCE_JSONL="${SOURCE_JSONL:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
STOCKFISH="${STOCKFISH:-$HOME/local/bin/stockfish}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed}"

LABEL_DEPTH="${LABEL_DEPTH:-16}"
LABEL_SHARDS="${LABEL_SHARDS:-24}"
LABEL_THREADS="${LABEL_THREADS:-1}"
LABEL_HASH="${LABEL_HASH:-128}"
LABEL_MAX_ABS_CP="${LABEL_MAX_ABS_CP:-1600}"
LABEL_PROGRESS="${LABEL_PROGRESS:-10000}"
SPRT_GAMES="${SPRT_GAMES:-4000}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_ELO1="${SPRT_ELO1:-8}"
SPRT_RESTART="${SPRT_RESTART:-off}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

label_pids=()
cleanup() {
  local rc=$?
  if [ "${#label_pids[@]}" -gt 0 ]; then
    kill "${label_pids[@]}" >/dev/null 2>&1 || true
  fi
  notify "Enyo NNUE d16 expansion finished rc=$rc run=$RUN"
  exit "$rc"
}
trap cleanup EXIT

notify "Enyo NNUE d16 expansion start run=$RUN"

for required in "$PY" "$TOOLS/sample_signed_buckets.py" "$TOOLS/label_with_uci.py" \
  "$TOOLS/pack_dataset.py" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" \
  "$SOURCE_JSONL" "$INIT" "$STOCKFISH" "$ENGINE" "$SPRT" "$BOOK" \
  "$D16_PACKED" "$SELFPLAY_PACKED" "$LICHESS_PACKED" "$BINPACK_PACKED"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE d16 expansion missing $required"
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

sample_source() {
  if [ -s "$RUN/source_signed.jsonl" ]; then
    return
  fi
  notify "Enyo NNUE d16 expansion: sample signed 1.5M from d12 pool"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$SOURCE_JSONL" \
    --output "$RUN/source_signed.jsonl" \
    --score-field score \
    --unique-fen \
    --seed 2026051501 \
    --progress 1000000 \
    --bucket any0:any:0:50:250000 \
    --bucket p50:pos:50:100:100000 \
    --bucket n50:neg:50:100:100000 \
    --bucket p100:pos:100:300:250000 \
    --bucket n100:neg:100:300:250000 \
    --bucket p300:pos:300:800:250000 \
    --bucket n300:neg:300:800:250000 \
    --bucket p800:pos:800:1600:125000 \
    --bucket n800:neg:800:1600:125000 | tee "$RUN/sample.log"
  local rows
  rows=$(wc -l < "$RUN/source_signed.jsonl" | tr -d ' ')
  notify "Enyo NNUE d16 expansion: sampled rows=$rows"
}

label_source() {
  if [ -s "$RUN/labeled.jsonl" ]; then
    return
  fi
  mkdir -p "$RUN/shards"
  notify "Enyo NNUE d16 expansion: Stockfish depth=$LABEL_DEPTH labels shards=$LABEL_SHARDS"
  for shard in $(seq 0 $((LABEL_SHARDS - 1))); do
    "$PY" "$TOOLS/label_with_uci.py" \
      --input "$RUN/source_signed.jsonl" \
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

  local total written last_pct alive now
  total=$(wc -l < "$RUN/source_signed.jsonl" | tr -d ' ')
  last_pct=-1
  while true; do
    written=$(find "$RUN/shards" -name 'label.*.jsonl' -type f -print0 \
      | xargs -0 cat 2>/dev/null | wc -l | tr -d ' ')
    now=$((written * 100 / total))
    if [ "$now" -ge $((last_pct + 10)) ] || [ "$now" -ge 99 ]; then
      notify "Enyo NNUE d16 expansion: label written=$written/$total (${now}%)"
      last_pct="$now"
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
  cat "$RUN"/shards/label.*.jsonl > "$RUN/labeled.jsonl"
  local rows
  rows=$(wc -l < "$RUN/labeled.jsonl" | tr -d ' ')
  notify "Enyo NNUE d16 expansion: labels ready rows=$rows"
}

pack_labeled() {
  if [ -s "$RUN/packed/meta.json" ]; then
    return
  fi
  notify "Enyo NNUE d16 expansion: pack labels"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$RUN/labeled.jsonl" \
    --out-dir "$RUN/packed" \
    --progress 200000 | tee "$RUN/pack.log"
}

train_candidate() {
  local tag="$1"
  local lr="$2"
  local epochs="$3"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ -s "$dir/model.nn" ]; then
    return
  fi
  notify "Enyo NNUE d16 expansion: train $tag"
  "$PY" "$TOOLS/train.py" \
    --data "$RUN/packed" \
    --init-from-nn "$INIT" \
    --objective huber \
    --huber-beta 200 \
    --select-metric mae \
    --epochs "$epochs" \
    --patience 2 \
    --batch-size 8192 \
    --lr "$lr" \
    --weight-decay 1e-6 \
    --target-clamp 800 \
    --device cuda \
    --workers 2 \
    --val-rows 100000 \
    --trainable all \
    --out "$dir/model.pt" \
    --out-nn "$dir/model.nn" | tee "$dir/train.log"
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
    --batch-size 8192 \
    --device cuda \
    --target-clamp 1600 \
    --buckets > "$out"
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  local own_rows own_skip d16_skip self_skip bin_skip
  own_rows=$(rows_in_meta "$RUN/packed")
  own_skip=$(eval_skip "$own_rows" 50000)
  d16_skip=$(eval_skip "$(rows_in_meta "$D16_PACKED")" 50000)
  self_skip=$(eval_skip "$(rows_in_meta "$SELFPLAY_PACKED")" 50000)
  bin_skip=$(eval_skip "$(rows_in_meta "$BINPACK_PACKED")" 50000)

  mkdir -p "$dir/eval"
  notify "Enyo NNUE d16 expansion: eval $tag"
  eval_one "$INIT" "$RUN/packed" "$own_skip" "$dir/eval/own_baseline.txt"
  eval_one "$dir/model.nn" "$RUN/packed" "$own_skip" "$dir/eval/own_candidate.txt"
  eval_one "$INIT" "$D16_PACKED" "$d16_skip" "$dir/eval/d16_baseline.txt"
  eval_one "$dir/model.nn" "$D16_PACKED" "$d16_skip" "$dir/eval/d16_candidate.txt"
  eval_one "$INIT" "$SELFPLAY_PACKED" "$self_skip" "$dir/eval/selfplay_baseline.txt"
  eval_one "$dir/model.nn" "$SELFPLAY_PACKED" "$self_skip" "$dir/eval/selfplay_candidate.txt"
  eval_one "$INIT" "$LICHESS_PACKED" 0 "$dir/eval/lichess_baseline.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" 0 "$dir/eval/lichess_candidate.txt"
  eval_one "$INIT" "$BINPACK_PACKED" "$bin_skip" "$dir/eval/binpack_baseline.txt"
  eval_one "$dir/model.nn" "$BINPACK_PACKED" "$bin_skip" "$dir/eval/binpack_candidate.txt"

  "$PY" - "$dir/eval" <<'PY' | tee "$dir/static_summary.txt"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
weights = {"own": 1.0, "d16": 1.0, "selfplay": 0.8, "lichess": 1.2, "binpack": 0.3}

def metric(path: pathlib.Path, key: str) -> float:
    text = path.read_text()
    m = re.search(rf"^{key}=([-+0-9.]+)%?$", text, re.M)
    if not m:
        raise SystemExit(f"missing {key} in {path}")
    return float(m.group(1))

score = 0.0
max_sign_drop = 0.0
print("dataset       weight base_mae cand_mae delta_mae base_sign cand_sign delta_sign")
for label, weight in weights.items():
    b = root / f"{label}_baseline.txt"
    c = root / f"{label}_candidate.txt"
    b_mae = metric(b, "mae")
    c_mae = metric(c, "mae")
    b_sign = metric(b, "sign")
    c_sign = metric(c, "sign")
    d_mae = b_mae - c_mae
    d_sign = c_sign - b_sign
    score += weight * d_mae
    max_sign_drop = max(max_sign_drop, -d_sign)
    print(
        f"{label:10} {weight:6.2f} {b_mae:8.3f} {c_mae:8.3f} {d_mae:9.3f}"
        f" {b_sign:9.2f} {c_sign:9.2f} {d_sign:10.2f}"
    )
print(f"score={score:.3f}")
print(f"max_sign_drop={max_sign_drop:.3f}")
PY
}

record_if_static_positive() {
  local tag="$1"
  local dir="$RUN/$tag"
  "$PY" - "$dir/static_summary.txt" "$tag" "$dir/model.nn" <<'PY' >> "$RUN/passed_candidates.tsv"
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
tag = sys.argv[2]
net = sys.argv[3]

def num(name: str) -> float:
    m = re.search(rf"^{name}=([-+0-9.]+)", text, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return float(m.group(1))

def delta(label: str, column: str) -> float:
    m = re.search(rf"^{label}\s+\S+\s+\S+\s+\S+\s+([-+0-9.]+)\s+\S+\s+\S+\s+([-+0-9.]+)", text, re.M)
    if not m:
        raise SystemExit(f"missing {label}")
    return float(m.group(1) if column == "mae" else m.group(2))

score = num("score")
max_sign_drop = num("max_sign_drop")
own_mae = delta("own", "mae")
d16_mae = delta("d16", "mae")
self_sign = delta("selfplay", "sign")
lichess_sign = delta("lichess", "sign")

passed = (
    score >= 5.0
    and max_sign_drop <= 0.40
    and own_mae > 1.0
    and d16_mae > 0.5
    and self_sign >= -0.30
    and lichess_sign >= -0.30
)
if passed:
    print(f"{score:.3f}\t{max_sign_drop:.3f}\t{tag}\t{net}")
PY
}

run_sprt_for_best() {
  if [ ! -s "$RUN/passed_candidates.tsv" ]; then
    notify "Enyo NNUE d16 expansion: no static-positive candidate; no SPRT"
    return
  fi

  local line tag net sprt_dir log final rc
  line=$(sort -k1,1nr "$RUN/passed_candidates.tsv" | head -1)
  tag=$(printf "%s" "$line" | cut -f3)
  net=$(printf "%s" "$line" | cut -f4)
  sprt_dir="$RUN/$tag/sprt"
  log="$sprt_dir/sprt.log"
  mkdir -p "$sprt_dir"

  notify "Enyo NNUE d16 expansion: ${SPRT_GAMES}-game SPRT start $tag"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "Hash=$SPRT_HASH" \
    --candidate-option "nnue_file=$net" \
    --reference-option "Hash=$SPRT_HASH" \
    --reference-option "nnue_file=$INIT" \
    --book "$BOOK" \
    --games "$SPRT_GAMES" \
    --elo0 0 \
    --elo1 "$SPRT_ELO1" \
    --concurrency "$SPRT_CONCURRENCY" \
    --threads "$SPRT_THREADS" \
    --tc "$SPRT_TC" \
    --restart "$SPRT_RESTART" \
    --log-dir "$sprt_dir" \
    --name "${tag}_vs_refnet" \
    --eta 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  final=$(rg --no-config "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" "$log" 2>/dev/null | tail -6 | tr '\n' ' ' || true)
  notify "Enyo NNUE d16 expansion: SPRT finished $tag rc=$rc ${final:-no final status}"
  return "$rc"
}

: > "$RUN/passed_candidates.tsv"

sample_source
label_source
pack_labeled

train_candidate d16exp_huber_cp800_lr1e6_e8 1e-6 8
eval_candidate d16exp_huber_cp800_lr1e6_e8
record_if_static_positive d16exp_huber_cp800_lr1e6_e8

train_candidate d16exp_huber_cp800_lr7e7_e8 7e-7 8
eval_candidate d16exp_huber_cp800_lr7e7_e8
record_if_static_positive d16exp_huber_cp800_lr7e7_e8

run_sprt_for_best
