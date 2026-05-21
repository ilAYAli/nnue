#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" >/dev/null 2>&1 || true
}

PY="${PY:-$HOME/.venv/bin/python}"
NNUE_REPO="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
ENGINE_REPO="${ENGINE_REPO:-$HOME/code/cpp/chess/enyo}"
TOOLS="$NNUE_REPO/tools/nnue"
RUN="${RUN:-$HOME/tmp/enyo_teacher/cp800_conservative_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

MIXED_PACKED="${MIXED_PACKED:-$HOME/tmp/enyo_teacher/source_mix_cp800_5m_20260514_080450/packed}"
D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

notify "Enyo NNUE cp800 conservative: start $RUN"

for required in \
  "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS/train.py" "$TOOLS/eval_dataset.py" \
  "$INIT" "$ENGINE" "$SPRT" "$BOOK" "$MIXED_PACKED" "$D16_PACKED" \
  "$SELFPLAY_PACKED" "$LICHESS_PACKED" "$BINPACK_PACKED"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE cp800 conservative: missing $required"
    exit 1
  fi
done

train_candidate() {
  local tag="$1"
  shift
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE cp800 conservative: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$MIXED_PACKED" \
      --init-from-nn "$INIT" \
      --epochs 4 \
      --patience 1 \
      --batch-size 8192 \
      --weight-decay 1e-6 \
      --device cuda \
      --workers 2 \
      --val-rows 50000 \
      --trainable all \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" \
      "$@" | tee "$dir/train.log"
  fi
  notify "Enyo NNUE cp800 conservative: train done $tag"
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
    --buckets \
    --sources > "$out"
}

eval_baseline() {
  mkdir -p "$RUN/baseline_eval"
  [ -s "$RUN/baseline_eval/d16.txt" ] \
    || eval_one "$INIT" "$D16_PACKED" 938632 "$RUN/baseline_eval/d16.txt"
  [ -s "$RUN/baseline_eval/selfplay.txt" ] \
    || eval_one "$INIT" "$SELFPLAY_PACKED" 20839426 "$RUN/baseline_eval/selfplay.txt"
  [ -s "$RUN/baseline_eval/lichess.txt" ] \
    || eval_one "$INIT" "$LICHESS_PACKED" 0 "$RUN/baseline_eval/lichess.txt"
  [ -s "$RUN/baseline_eval/binpack.txt" ] \
    || eval_one "$INIT" "$BINPACK_PACKED" 4900000 "$RUN/baseline_eval/binpack.txt"
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  mkdir -p "$dir/eval"
  notify "Enyo NNUE cp800 conservative: eval $tag"
  eval_baseline
  eval_one "$dir/model.nn" "$D16_PACKED" 938632 "$dir/eval/d16_candidate.txt"
  eval_one "$dir/model.nn" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay_candidate.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" 0 "$dir/eval/lichess_candidate.txt"
  eval_one "$dir/model.nn" "$BINPACK_PACKED" 4900000 "$dir/eval/binpack_candidate.txt"

  "$PY" - "$RUN/baseline_eval" "$dir/eval" <<'PY' | tee "$dir/static_summary.txt"
import pathlib
import re
import sys

baseline = pathlib.Path(sys.argv[1])
candidate = pathlib.Path(sys.argv[2])
weights = {
    "d16": 1.35,
    "selfplay": 0.75,
    "lichess": 1.25,
    "binpack": 0.35,
}

def metric(path: pathlib.Path, key: str) -> float:
    text = path.read_text()
    match = re.search(rf"^{key}=([-+0-9.]+)%?$", text, re.M)
    if not match:
        raise SystemExit(f"missing {key} in {path}")
    return float(match.group(1))

score = 0.0
max_sign_drop = 0.0
print("dataset       weight base_mae cand_mae delta_mae base_sign cand_sign delta_sign")
for label, weight in weights.items():
    b = baseline / f"{label}.txt"
    c = candidate / f"{label}_candidate.txt"
    b_mae = metric(b, "mae")
    c_mae = metric(c, "mae")
    b_sign = metric(b, "sign")
    c_sign = metric(c, "sign")
    d_mae = b_mae - c_mae
    d_sign = c_sign - b_sign
    max_sign_drop = max(max_sign_drop, -d_sign)
    score += weight * d_mae
    print(
        f"{label:10} {weight:6.2f} {b_mae:8.3f} {c_mae:8.3f} {d_mae:9.3f}"
        f" {b_sign:9.2f} {c_sign:9.2f} {d_sign:10.2f}"
    )
print(f"score={score:.3f}")
print(f"max_sign_drop={max_sign_drop:.3f}")
PY
}

record_candidate_if_passed() {
  local tag="$1"
  local dir="$RUN/$tag"
  "$PY" - "$tag" "$dir/model.nn" "$dir/static_summary.txt" <<'PY' >> "$RUN/passed_candidates.tsv"
import pathlib
import re
import sys

tag, net, summary_path = sys.argv[1:]
summary = pathlib.Path(summary_path).read_text()
score = float(re.search(r"^score=([-+0-9.]+)$", summary, re.M).group(1))
max_sign_drop = float(re.search(r"^max_sign_drop=([-+0-9.]+)$", summary, re.M).group(1))
delta = {}
for line in summary.splitlines():
    if line.startswith(("d16", "selfplay", "lichess", "binpack")):
        parts = line.split()
        delta[parts[0]] = (float(parts[4]), float(parts[7]))

passed = (
    score > 0.75
    and max_sign_drop <= 0.08
    and delta["d16"][0] > 0.10
    and delta["lichess"][0] > 0.10
    and delta["selfplay"][1] > -0.10
    and delta["binpack"][1] > -0.10
)
if passed:
    print(f"{score:.3f}\t{tag}\t{net}")
PY
}

latest_sprt_line() {
  local log="$1"
  grep -E "^\[[[:space:]]*[0-9]+/[0-9]+\]" "$log" 2>/dev/null | tail -1 || true
}

monitor_sprt() {
  local log="$1"
  local pid="$2"
  local tag="$3"
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep 60
    local line games elo llr
    line=$(latest_sprt_line "$log")
    [ -n "$line" ] || continue
    games=$(printf "%s\n" "$line" | sed -n "s/^\[[[:space:]]*\([0-9][0-9]*\)\/[0-9][0-9]*\].*/\1/p")
    elo=$(printf "%s\n" "$line" | sed -n "s/.*Elo[[:space:]]*\([-+0-9.][-.+0-9]*\)[[:space:]]*+\/.*/\1/p")
    llr=$(printf "%s\n" "$line" | sed -n "s/.*LLR[[:space:]]*\([-+0-9.][-.+0-9]*\)\/2\.94.*/\1/p")
    [ -n "$games" ] && [ -n "$elo" ] && [ -n "$llr" ] || continue
    if awk -v g="$games" -v e="$elo" -v l="$llr" \
      'BEGIN { exit !(g >= 800 && (e < 3.0 || l < 0.10)) }'
    then
      notify "Enyo NNUE cp800 conservative: stop $tag at ${games}: Elo ${elo}, LLR ${llr}"
      kill -INT "$pid" >/dev/null 2>&1 || true
      sleep 5
      pkill -P "$pid" >/dev/null 2>&1 || true
      return
    fi
    if awk -v g="$games" -v e="$elo" -v l="$llr" \
      'BEGIN { exit !(g >= 1400 && (e < 6.0 || l < 0.35)) }'
    then
      notify "Enyo NNUE cp800 conservative: stop $tag at ${games}: Elo ${elo}, LLR ${llr}"
      kill -INT "$pid" >/dev/null 2>&1 || true
      sleep 5
      pkill -P "$pid" >/dev/null 2>&1 || true
      return
    fi
  done
}

run_sprt_for_best() {
  if [ ! -s "$RUN/passed_candidates.tsv" ]; then
    notify "Enyo NNUE cp800 conservative: no candidate passed static gate"
    return 0
  fi
  local best_line score tag net log sprt_pid monitor_pid rc status
  best_line=$(sort -k1,1nr "$RUN/passed_candidates.tsv" | head -1)
  score=$(printf "%s" "$best_line" | cut -f1)
  tag=$(printf "%s" "$best_line" | cut -f2)
  net=$(printf "%s" "$best_line" | cut -f3)
  mkdir -p "$RUN/sprt"
  notify "Enyo NNUE cp800 conservative: 4000-game SPRT start $tag static_score=$score"
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "nnue_file=$net" \
    --candidate-option "Hash=1024" \
    --reference-option "nnue_file=$INIT" \
    --reference-option "Hash=1024" \
    --book "$BOOK" \
    --games 4000 \
    --elo0 0 \
    --elo1 5 \
    --concurrency 6 \
    --threads 4 \
    --tc "10+0.1" \
    --log-dir "$RUN/sprt" \
    --name "${tag}_vs_refnet_e5" &
  sprt_pid=$!
  log="$RUN/sprt/${tag}_vs_refnet_e5.log"
  monitor_sprt "$log" "$sprt_pid" "$tag" &
  monitor_pid=$!
  set +e
  wait "$sprt_pid"
  rc=$?
  set -e
  kill "$monitor_pid" >/dev/null 2>&1 || true
  status=$(latest_sprt_line "$log")
  notify "Enyo NNUE cp800 conservative: SPRT finished $tag rc=$rc ${status:-no final status}"
  return "$rc"
}

: > "$RUN/passed_candidates.tsv"

train_candidate cp800_cons_huber_lr1e7_e4 \
  --objective huber \
  --huber-beta 200 \
  --select-metric mae \
  --target-clamp 800 \
  --lr 1e-7
eval_candidate cp800_cons_huber_lr1e7_e4
record_candidate_if_passed cp800_cons_huber_lr1e7_e4

train_candidate cp800_cons_huber_lr3e8_e4 \
  --objective huber \
  --huber-beta 200 \
  --select-metric mae \
  --target-clamp 800 \
  --lr 3e-8
eval_candidate cp800_cons_huber_lr3e8_e4
record_candidate_if_passed cp800_cons_huber_lr3e8_e4

train_candidate cp800_cons_mpe25_lr1e7_e4 \
  --objective mpe25 \
  --select-metric loss \
  --wdl-lambda 0.75 \
  --target-clamp 800 \
  --lr 1e-7
eval_candidate cp800_cons_mpe25_lr1e7_e4
record_candidate_if_passed cp800_cons_mpe25_lr1e7_e4

run_sprt_for_best
rc=$?
notify "Enyo NNUE cp800 conservative: finished rc=$rc run=$RUN"
exit "$rc"
