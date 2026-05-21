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
SOURCE_RUN="${SOURCE_RUN:-$HOME/tmp/enyo_teacher/d16_expansion_20260515_032144}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/d16_mpe_sweep_$(date +%Y%m%d_%H%M%S)}"
PACKED="${PACKED:-$SOURCE_RUN/packed}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed}"

SPRT_GAMES="${SPRT_GAMES:-4000}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_ELO1="${SPRT_ELO1:-8}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

trap 'rc=$?; notify "Enyo NNUE d16 MPE sweep finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE d16 MPE sweep start run=$RUN"

for required in "$PY" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" "$PACKED" \
  "$INIT" "$ENGINE" "$SPRT" "$BOOK" "$D16_PACKED" "$SELFPLAY_PACKED" \
  "$LICHESS_PACKED" "$BINPACK_PACKED"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE d16 MPE sweep missing $required"
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

train_candidate() {
  local tag="$1"
  local wdl="$2"
  local lr="$3"
  local clamp="$4"
  local epochs="$5"
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  notify "Enyo NNUE d16 MPE sweep: train $tag"
  "$PY" "$TOOLS/train.py" \
    --data "$PACKED" \
    --init-from-nn "$INIT" \
    --objective mpe25 \
    --wdl-lambda "$wdl" \
    --select-metric loss \
    --epochs "$epochs" \
    --patience 2 \
    --batch-size 8192 \
    --lr "$lr" \
    --weight-decay 1e-6 \
    --target-clamp "$clamp" \
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
  local own_skip d16_skip self_skip bin_skip
  own_skip=$(eval_skip "$(rows_in_meta "$PACKED")")
  d16_skip=$(eval_skip "$(rows_in_meta "$D16_PACKED")")
  self_skip=$(eval_skip "$(rows_in_meta "$SELFPLAY_PACKED")")
  bin_skip=$(eval_skip "$(rows_in_meta "$BINPACK_PACKED")")

  mkdir -p "$dir/eval"
  notify "Enyo NNUE d16 MPE sweep: eval $tag"
  eval_one "$INIT" "$PACKED" "$own_skip" "$dir/eval/own_baseline.txt"
  eval_one "$dir/model.nn" "$PACKED" "$own_skip" "$dir/eval/own_candidate.txt"
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

record_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  "$PY" - "$dir/static_summary.txt" "$tag" "$dir/model.nn" <<'PY' >> "$RUN/candidates.tsv"
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
tag = sys.argv[2]
net = sys.argv[3]

def num(name):
    m = re.search(rf"^{name}=([-+0-9.]+)", text, re.M)
    if not m:
        raise SystemExit(f"missing {name}")
    return float(m.group(1))

score = num("score")
max_sign_drop = num("max_sign_drop")
passed = score >= 3.0 and max_sign_drop <= 0.5
print(f"{int(passed)}\t{score:.3f}\t{max_sign_drop:.3f}\t{tag}\t{net}")
PY
}

run_sprt_for_best() {
  local line passed tag net sprt_dir log final rc
  line=$(sort -k1,1nr -k2,2nr "$RUN/candidates.tsv" | head -1)
  passed=$(printf "%s" "$line" | cut -f1)
  tag=$(printf "%s" "$line" | cut -f4)
  net=$(printf "%s" "$line" | cut -f5)
  if [ "$passed" != "1" ]; then
    notify "Enyo NNUE d16 MPE sweep: no static-positive candidate; no SPRT"
    return 0
  fi
  sprt_dir="$RUN/$tag/sprt"
  log="$sprt_dir/sprt.log"
  mkdir -p "$sprt_dir"
  notify "Enyo NNUE d16 MPE sweep: ${SPRT_GAMES}-game SPRT start $tag"
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
    --restart off \
    --log-dir "$sprt_dir" \
    --name "${tag}_vs_refnet" \
    --eta 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  final=$(rg --no-config "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" "$log" 2>/dev/null | tail -6 | tr '\n' ' ' || true)
  notify "Enyo NNUE d16 MPE sweep: SPRT finished $tag rc=$rc ${final:-no final status}"
  return "$rc"
}

: > "$RUN/candidates.tsv"

train_candidate mpe_wdl095_lr3e7_cp1200_e4 0.95 3e-7 1200 4
eval_candidate mpe_wdl095_lr3e7_cp1200_e4
record_candidate mpe_wdl095_lr3e7_cp1200_e4

train_candidate mpe_wdl075_lr3e7_cp1200_e4 0.75 3e-7 1200 4
eval_candidate mpe_wdl075_lr3e7_cp1200_e4
record_candidate mpe_wdl075_lr3e7_cp1200_e4

train_candidate mpe_wdl095_lr1e7_cp1200_e4 0.95 1e-7 1200 4
eval_candidate mpe_wdl095_lr1e7_cp1200_e4
record_candidate mpe_wdl095_lr1e7_cp1200_e4

run_sprt_for_best
