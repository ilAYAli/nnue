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
TOOLS="$NNUE_REPO/tools/nnue2"
SOURCE_RUN="${SOURCE_RUN:-$(ls -td "$HOME"/tmp/enyo_teacher/hardcase_aug_* 2>/dev/null | head -1)}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/hardcase_pairwise_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
BASE_INIT="${BASE_INIT:-$SOURCE_RUN/hardmajor_huber_lr3e6_e8/model.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"
PACKED="${PACKED:-$SOURCE_RUN/hardmajor_d16500000_bin350000_hard150000/packed}"
PAIRS="${PAIRS:-$SOURCE_RUN/hard_child_labeled.jsonl}"
CASES="${CASES:-$SOURCE_RUN/hard_cases.jsonl}"
D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed}"
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
trap 'rc=$?; notify "Enyo NNUE hardcase pairwise sweep finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE hardcase pairwise sweep start run=$RUN source=$SOURCE_RUN"

for required in "$PY" "$TOOLS/train_pairwise.py" "$TOOLS/eval_dataset.py" \
  "$TOOLS/eval_move_gate.py" "$INIT" "$BASE_INIT" "$ENGINE" "$SPRT" "$BOOK" \
  "$PACKED" "$PAIRS" "$CASES" "$D16_PACKED" "$SELFPLAY_PACKED" \
  "$LICHESS_PACKED" "$BINPACK_PACKED"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE hardcase pairwise sweep missing $required"
    exit 1
  fi
done

train_candidate() {
  local tag="$1"
  local pair_weight="$2"
  local lr="$3"
  local epochs="$4"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  notify "Enyo NNUE hardcase pairwise sweep: train $tag"
  "$PY" "$TOOLS/train_pairwise.py" \
    --data "$PACKED" \
    --pairs "$PAIRS" \
    --init-from-nn "$BASE_INIT" \
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
  notify "Enyo NNUE hardcase pairwise sweep: eval $tag"
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

hard_gate_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  notify "Enyo NNUE hardcase pairwise sweep: hard gate $tag"
  "$PY" "$TOOLS/eval_move_gate.py" \
    --cases "$CASES" \
    --engine "$ENGINE" \
    --baseline-net "$INIT" \
    --candidate-net "$dir/model.nn" \
    --output "$dir/hard_gate.jsonl" | tee "$dir/hard_gate.txt"

  "$PY" - "$dir/static_summary.txt" "$dir/hard_gate.txt" "$tag" "$dir/model.nn" <<'PY' >> "$RUN/passed_candidates.tsv"
import pathlib
import re
import sys

static = pathlib.Path(sys.argv[1]).read_text()
gate = pathlib.Path(sys.argv[2]).read_text()
tag = sys.argv[3]
net = sys.argv[4]

def num(text: str, name: str) -> float:
    m = re.search(rf"^{name}=([-+0-9.]+)", text, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return float(m.group(1))

def pair(name: str) -> tuple[int, int]:
    m = re.search(rf"^{name}=([0-9]+)/([0-9]+)", gate, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return int(m.group(1)), int(m.group(2))

score = num(static, "score")
max_sign_drop = num(static, "max_sign_drop")
fixed = int(num(gate, "fixed"))
regressed = int(num(gate, "regressed"))
better, total = pair("candidate_better_margin")
weighted = num(gate, "delta_loss_weighted_margin")
avg = num(gate, "delta_avg_margin")

passed = (
    score >= 5.0
    and max_sign_drop <= 0.5
    and weighted >= 3.0
    and avg >= 2.0
    and better >= total // 2 + 1
    and fixed >= regressed
)
if passed:
    rank = score + 4.0 * weighted + 2.0 * avg + fixed - regressed
    print(f"{rank:.3f}\t{score:.3f}\t{weighted:.3f}\t{avg:.3f}\t{tag}\t{net}")
PY
}

run_sprt_for_best() {
  if [ ! -s "$RUN/passed_candidates.tsv" ]; then
    notify "Enyo NNUE hardcase pairwise sweep: no gate-positive candidate; no SPRT"
    exit 0
  fi
  local line tag net sprt_dir log status
  line=$(sort -k1,1nr "$RUN/passed_candidates.tsv" | head -1)
  tag=$(printf "%s" "$line" | cut -f5)
  net=$(printf "%s" "$line" | cut -f6)
  sprt_dir="$RUN/sprt"
  log="$sprt_dir/${tag}_vs_refnet.log"
  mkdir -p "$sprt_dir"
  notify "Enyo NNUE hardcase pairwise sweep: ${SPRT_GAMES}-game SPRT start $tag"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "Hash=$SPRT_HASH" \
    --candidate-option "nnue2_file=$net" \
    --reference-option "Hash=$SPRT_HASH" \
    --reference-option "nnue2_file=$INIT" \
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
  local rc=${PIPESTATUS[0]}
  set -e
  status=$(rg --no-config "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" "$log" | tail -6 | tr '\n' ' ')
  notify "Enyo NNUE hardcase pairwise sweep: SPRT finished $tag rc=$rc $status"
  exit "$rc"
}

: > "$RUN/passed_candidates.tsv"

train_candidate pair_w10_lr3e6_e8 10 3e-6 8
eval_candidate pair_w10_lr3e6_e8
hard_gate_candidate pair_w10_lr3e6_e8

train_candidate pair_w30_lr3e6_e8 30 3e-6 8
eval_candidate pair_w30_lr3e6_e8
hard_gate_candidate pair_w30_lr3e6_e8

train_candidate pair_w100_lr3e6_e8 100 3e-6 8
eval_candidate pair_w100_lr3e6_e8
hard_gate_candidate pair_w100_lr3e6_e8

train_candidate pair_w30_lr1e5_e6 30 1e-5 6
eval_candidate pair_w30_lr1e5_e6
hard_gate_candidate pair_w30_lr1e5_e6

run_sprt_for_best
