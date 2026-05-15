#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
ENGINE_REPO="${ENGINE_REPO:-$HOME/code/cpp/chess/enyo}"
TOOLS="$NNUE_REPO/tools/nnue2"
RUN="${RUN:-$HOME/tmp/enyo_teacher/hardcase_aug_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
BUGS="${BUGS:-$ENGINE_REPO/bugs}"
if [ -z "${STOCKFISH:-}" ]; then
  for candidate in \
    /usr/games/stockfish \
    "$HOME/local/bin/stockfish" \
    "$HOME/local/stockfish-18/stockfish-ubuntu-x86-64-avx2" \
    "$(command -v stockfish 2>/dev/null || true)"
  do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      STOCKFISH="$candidate"
      break
    fi
  done
fi
STOCKFISH="${STOCKFISH:-/usr/games/stockfish}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"
D16_JSON="${D16_JSON:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl}"
BINPACK_JSON="${BINPACK_JSON:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/binpack.jsonl}"
D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed}"
HARD_MIN_LOSS="${HARD_MIN_LOSS:-150}"
HARD_MAX_LOSS="${HARD_MAX_LOSS:-999}"
HARD_REPEAT="${HARD_REPEAT:-10000}"
HARD_ROWS="${HARD_ROWS:-150000}"
D16_ROWS="${D16_ROWS:-500000}"
BINPACK_ROWS="${BINPACK_ROWS:-350000}"
HARD_SOURCE_WEIGHT="${HARD_SOURCE_WEIGHT:-5.0}"
TEACHER_DEPTH="${TEACHER_DEPTH:-20}"
SPRT_GAMES="${SPRT_GAMES:-1000}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_ELO1="${SPRT_ELO1:-8}"
SPRT_RESTART="${SPRT_RESTART:-off}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE hardcase aug finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE hardcase aug: start $RUN"

for required in \
  "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS/build_move_gate.py" \
  "$TOOLS/move_gate_to_child_rows.py" "$TOOLS/repeat_jsonl.py" \
  "$TOOLS/label_with_uci.py" "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" \
  "$TOOLS/train.py" "$TOOLS/train_pairwise.py" "$TOOLS/eval_dataset.py" "$TOOLS/eval_move_gate.py" \
  "$INIT" "$ENGINE" "$BUGS" "$STOCKFISH" "$SPRT" "$BOOK" \
  "$D16_JSON" "$BINPACK_JSON" "$D16_PACKED" "$SELFPLAY_PACKED" \
  "$LICHESS_PACKED" "$BINPACK_PACKED"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE hardcase aug: missing $required"
    exit 1
  fi
done

prepare_hard_rows() {
  if [ ! -s "$RUN/hard_cases.jsonl" ]; then
    notify "Enyo NNUE hardcase aug: build move gate cases"
    "$PY" "$TOOLS/build_move_gate.py" "$BUGS" \
      --output "$RUN/hard_cases.jsonl" \
      --min-loss "$HARD_MIN_LOSS" \
      --max-loss "$HARD_MAX_LOSS"
  fi
  if [ ! -s "$RUN/hard_child_rows.jsonl" ]; then
    notify "Enyo NNUE hardcase aug: expand hard cases to child rows"
    "$PY" "$TOOLS/move_gate_to_child_rows.py" \
      --cases "$RUN/hard_cases.jsonl" \
      --output "$RUN/hard_child_rows.jsonl"
  fi
  if [ ! -s "$RUN/hard_child_labeled.jsonl" ]; then
    notify "Enyo NNUE hardcase aug: Stockfish d$TEACHER_DEPTH labels for hard children"
    "$PY" "$TOOLS/label_with_uci.py" \
      --input "$RUN/hard_child_rows.jsonl" \
      --output "$RUN/hard_child_labeled.jsonl" \
      --engine "$STOCKFISH" \
      --depth "$TEACHER_DEPTH" \
      --threads 4 \
      --hash 512 \
      --max-abs-cp 1600 \
      --progress 10
  fi
  if [ ! -s "$RUN/hard_child_labeled_x${HARD_REPEAT}.jsonl" ]; then
    notify "Enyo NNUE hardcase aug: repeat hard labels x$HARD_REPEAT"
    "$PY" "$TOOLS/repeat_jsonl.py" \
      --input "$RUN/hard_child_labeled.jsonl" \
      --output "$RUN/hard_child_labeled_x${HARD_REPEAT}.jsonl" \
      --repeat "$HARD_REPEAT"
  fi
}

mix_and_pack() {
  local dir="$RUN/hardmajor_d16${D16_ROWS}_bin${BINPACK_ROWS}_hard${HARD_ROWS}"
  mkdir -p "$dir"
  if [ ! -s "$dir/source.jsonl" ]; then
    notify "Enyo NNUE hardcase aug: mix d16/binpack/major-hard"
    "$PY" "$TOOLS/mix_jsonl.py" \
      --output "$dir/source.jsonl" \
      --source "$D16_JSON:$D16_ROWS" \
      --source "$BINPACK_JSON:$BINPACK_ROWS" \
      --source "$RUN/hard_child_labeled_x${HARD_REPEAT}.jsonl:$HARD_ROWS" \
      --seed 2026051401 \
      --progress 200000
  fi
  if [ ! -s "$dir/packed/meta.json" ]; then
    notify "Enyo NNUE hardcase aug: pack hard mix"
    "$PY" "$TOOLS/pack_dataset.py" \
      --input "$dir/source.jsonl" \
      --out-dir "$dir/packed" \
      --progress 200000
  fi
}

train_candidate() {
  local tag="$1"
  local lr="$2"
  local src="$RUN/hardmajor_d16${D16_ROWS}_bin${BINPACK_ROWS}_hard${HARD_ROWS}"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE hardcase aug: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$src/packed" \
      --init-from-nn "$INIT" \
      --objective huber \
      --huber-beta 200 \
      --select-metric mae \
      --epochs 8 \
      --patience 2 \
      --batch-size 8192 \
      --lr "$lr" \
      --weight-decay 1e-6 \
      --target-clamp 1600 \
      --device cuda \
      --workers 2 \
      --val-rows 50000 \
      --source-loss-weight "hard_move_child=$HARD_SOURCE_WEIGHT" \
      --trainable all \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" | tee "$dir/train.log"
  fi
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
  mkdir -p "$dir/eval"
  notify "Enyo NNUE hardcase aug: eval $tag"
  eval_one "$INIT" "$D16_PACKED" 938632 "$dir/eval/d16_baseline.txt"
  eval_one "$dir/model.nn" "$D16_PACKED" 938632 "$dir/eval/d16_candidate.txt"
  eval_one "$INIT" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay_baseline.txt"
  eval_one "$dir/model.nn" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay_candidate.txt"
  eval_one "$INIT" "$LICHESS_PACKED" 0 "$dir/eval/lichess_baseline.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" 0 "$dir/eval/lichess_candidate.txt"
  eval_one "$INIT" "$BINPACK_PACKED" 4900000 "$dir/eval/binpack_baseline.txt"
  eval_one "$dir/model.nn" "$BINPACK_PACKED" 4900000 "$dir/eval/binpack_candidate.txt"

  "$PY" - "$dir/eval" <<'PY' | tee "$dir/static_summary.txt"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
weights = {"d16": 1.2, "selfplay": 0.8, "lichess": 1.2, "binpack": 0.3}

def metric(path: pathlib.Path, key: str) -> float:
    text = path.read_text()
    m = re.search(rf"^{key}=([-+0-9.]+)%?$", text, re.M)
    if not m:
        raise SystemExit(f"missing {key} in {path}")
    return float(m.group(1))

score = 0.0
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
    print(
        f"{label:10} {weight:6.2f} {b_mae:8.3f} {c_mae:8.3f} {d_mae:9.3f}"
        f" {b_sign:9.2f} {c_sign:9.2f} {d_sign:10.2f}"
    )
print(f"score={score:.3f}")
PY
}

hard_gate_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  notify "Enyo NNUE hardcase aug: hard gate $tag"
  "$PY" "$TOOLS/eval_move_gate.py" \
    --cases "$RUN/hard_cases.jsonl" \
    --engine "$ENGINE" \
    --baseline-net "$INIT" \
    --candidate-net "$dir/model.nn" \
    --output "$dir/hard_gate.jsonl" | tee "$dir/hard_gate.txt"

  "$PY" - "$dir/hard_gate.txt" "$tag" "$dir/model.nn" <<'PY' >> "$RUN/passed_candidates.tsv"
import pathlib
import re
import sys

summary = pathlib.Path(sys.argv[1]).read_text()
tag = sys.argv[2]
net = sys.argv[3]

def num(name: str) -> float:
    m = re.search(rf"^{name}=([-+0-9.]+)", summary, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return float(m.group(1))

def pair(name: str) -> tuple[int, int]:
    m = re.search(rf"^{name}=([0-9]+)/([0-9]+)", summary, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return int(m.group(1)), int(m.group(2))

fixed = int(num("fixed"))
regressed = int(num("regressed"))
better, total = pair("candidate_better_margin")
weighted = num("delta_loss_weighted_margin")
avg = num("delta_avg_margin")
min_better = total // 2 + 1
passed = (
    weighted >= 3.0
    and avg >= 1.0
    and better >= min_better
    and fixed >= regressed
)
if passed:
    print(f"{weighted:.3f}\t{avg:.3f}\t{tag}\t{net}")
PY
}

train_pairwise_candidate() {
  local tag="$1"
  local init="$2"
  local lr="$3"
  local pair_weight="$4"
  local epochs="$5"
  local src="$RUN/hardmajor_d16${D16_ROWS}_bin${BINPACK_ROWS}_hard${HARD_ROWS}"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE hardcase aug: pairwise train $tag"
    "$PY" "$TOOLS/train_pairwise.py" \
      --data "$src/packed" \
      --pairs "$RUN/hard_child_labeled.jsonl" \
      --init-from-nn "$init" \
      --epochs "$epochs" \
      --batch-size 8192 \
      --pair-batch-size 64 \
      --lr "$lr" \
      --weight-decay 1e-6 \
      --huber-beta 200 \
      --pair-beta 100 \
      --pair-weight "$pair_weight" \
      --target-clamp 1600 \
      --max-target-margin 800 \
      --min-target-margin 1 \
      --loss-weight-by-cp \
      --device cuda \
      --workers 2 \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" | tee "$dir/train.log"
  fi
}

run_sprt_for_best() {
  if [ ! -s "$RUN/passed_candidates.tsv" ]; then
    notify "Enyo NNUE hardcase aug: no hard-gate-positive candidate; no SPRT"
    exit 0
  fi
  local best_line weighted avg tag net
  best_line=$(sort -k1,1nr "$RUN/passed_candidates.tsv" | head -1)
  weighted=$(printf "%s" "$best_line" | cut -f1)
  avg=$(printf "%s" "$best_line" | cut -f2)
  tag=$(printf "%s" "$best_line" | cut -f3)
  net=$(printf "%s" "$best_line" | cut -f4)
  notify "Enyo NNUE hardcase aug: ${SPRT_GAMES}-game SPRT start $tag hard_weighted=$weighted avg=$avg"
  mkdir -p "$RUN/sprt"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "nnue2_file=$net" \
    --candidate-option "Hash=$SPRT_HASH" \
    --reference-option "nnue2_file=$INIT" \
    --reference-option "Hash=$SPRT_HASH" \
    --book "$BOOK" \
    --games "$SPRT_GAMES" \
    --elo0 0 \
    --elo1 "$SPRT_ELO1" \
    --concurrency "$SPRT_CONCURRENCY" \
    --threads "$SPRT_THREADS" \
    --tc "$SPRT_TC" \
    --restart "$SPRT_RESTART" \
    --log-dir "$RUN/sprt" \
    --name "${tag}_vs_reference_net" \
    --eta | tee "$RUN/sprt/${tag}_vs_reference_net.log"
  local sprt_rc=$?
  set -e
  local log="$RUN/sprt/${tag}_vs_reference_net.log"
  local final_status
  final_status=$(rg --no-config "^(\\[[ 0-9]+/|Score of|Elo difference:|SPRT:|Finished match|Total Time)" "$log" 2>/dev/null | tail -6 | tr "\n" " " || true)
  notify "Enyo NNUE hardcase aug: SPRT finished $tag rc=$sprt_rc ${final_status:-no final status}"
  exit "$sprt_rc"
}

: > "$RUN/passed_candidates.tsv"
prepare_hard_rows
mix_and_pack

train_candidate "hardmajor_huber_lr1e6_e8" "1e-6"
eval_candidate "hardmajor_huber_lr1e6_e8"
hard_gate_candidate "hardmajor_huber_lr1e6_e8"

train_candidate "hardmajor_huber_lr3e6_e8" "3e-6"
eval_candidate "hardmajor_huber_lr3e6_e8"
hard_gate_candidate "hardmajor_huber_lr3e6_e8"

train_pairwise_candidate "hardpair_from_init_w1_lr1e6_e4" "$INIT" "1e-6" "1.0" 4
eval_candidate "hardpair_from_init_w1_lr1e6_e4"
hard_gate_candidate "hardpair_from_init_w1_lr1e6_e4"

train_pairwise_candidate "hardpair_from_huber3_w1_lr3e7_e4" "$RUN/hardmajor_huber_lr3e6_e8/model.nn" "3e-7" "1.0" 4
eval_candidate "hardpair_from_huber3_w1_lr3e7_e4"
hard_gate_candidate "hardpair_from_huber3_w1_lr3e7_e4"

run_sprt_for_best
