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
RUN="${RUN:-$HOME/tmp/enyo_teacher/nonzero_binpack_sweep_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

BINPACK_FILE="${BINPACK_FILE:-$HOME/code/cpp/chess/assets/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack}"
D16_JSON="${D16_JSON:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl}"
LICHESS_JSON="${LICHESS_JSON:-$HOME/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl}"
D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"

BINPACK_ROWS="${BINPACK_ROWS:-1600000}"
BINPACK_MIN_ABS_CP="${BINPACK_MIN_ABS_CP:-50}"
BINPACK_MAX_ABS_CP="${BINPACK_MAX_ABS_CP:-1600}"
SPRT_GAMES="${SPRT_GAMES:-1000}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_ELO1="${SPRT_ELO1:-8}"
SPRT_RESTART="${SPRT_RESTART:-off}"
ONLY_SPRT_TAG="${ONLY_SPRT_TAG:-}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

notify "Enyo NNUE nonzero-binpack sweep: start $RUN"

for required in \
  "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS/binpack_to_jsonl.cpp" \
  "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" "$TOOLS/train.py" \
  "$TOOLS/eval_dataset.py" "$INIT" "$ENGINE" "$SPRT" "$BOOK" \
  "$BINPACK_FILE" "$D16_JSON" "$LICHESS_JSON" \
  "$D16_PACKED" "$LICHESS_PACKED" "$SELFPLAY_PACKED"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE nonzero-binpack sweep: missing $required"
    exit 1
  fi
done

BINPACK_CONVERTER="$RUN/binpack_to_jsonl"
BINPACK_JSON="$RUN/binpack_nonzero.jsonl"
BINPACK_PACKED="$RUN/binpack_nonzero_packed"
MIXED_JSON="$RUN/source.jsonl"
MIXED_PACKED="$RUN/packed"

if [ ! -x "$BINPACK_CONVERTER" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: build binpack converter"
  c++ -O3 -std=c++23 \
    -I "$HOME/tmp/nnue-pytorch/data_loader/cpp/lib" \
    "$TOOLS/binpack_to_jsonl.cpp" \
    -lfmt \
    -o "$BINPACK_CONVERTER"
fi

if [ ! -s "$BINPACK_JSON" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: convert $BINPACK_ROWS nonzero binpack rows"
  "$BINPACK_CONVERTER" \
    --input "$BINPACK_FILE" \
    --output "$BINPACK_JSON" \
    --limit "$BINPACK_ROWS" \
    --min-abs-cp "$BINPACK_MIN_ABS_CP" \
    --max-abs-cp "$BINPACK_MAX_ABS_CP" \
    --progress 250000 | tee "$RUN/binpack_convert.log"
fi

if [ ! -s "$BINPACK_PACKED/meta.json" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: pack nonzero binpack rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$BINPACK_JSON" \
    --out-dir "$BINPACK_PACKED" \
    --progress 250000 | tee "$RUN/binpack_pack.log"
fi

if [ ! -s "$MIXED_JSON" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: mix d16 + lichess + nonzero binpack"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIXED_JSON" \
    --source "$D16_JSON:900000" \
    --source "$LICHESS_JSON:2500000" \
    --source "$BINPACK_JSON:1600000" \
    --seed 2026051403 \
    --progress 500000 | tee "$RUN/mix.log"
fi

if [ ! -s "$MIXED_PACKED/meta.json" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: pack mixed rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$MIXED_JSON" \
    --out-dir "$MIXED_PACKED" \
    --progress 500000 | tee "$RUN/pack.log"
fi

train_candidate() {
  local tag="$1"
  local objective="$2"
  local lr="$3"
  shift 3
  local dir="$RUN/$tag"
  mkdir -p "$dir"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE nonzero-binpack sweep: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$MIXED_PACKED" \
      --init-from-nn "$INIT" \
      --objective "$objective" \
      --huber-beta 200 \
      --select-metric mae \
      --epochs 8 \
      --patience 2 \
      --batch-size 8192 \
      --lr "$lr" \
      --weight-decay 1e-6 \
      --target-clamp 1200 \
      --device cuda \
      --workers 2 \
      --val-rows 100000 \
      --trainable all \
      --source-loss-weight stockfish=1.25 \
      --source-loss-weight lichess_eval=1.0 \
      --source-loss-weight stockfish_binpack=0.35 \
      --source-wdl-lambda stockfish=1.0 \
      --source-wdl-lambda lichess_eval=1.0 \
      --source-wdl-lambda stockfish_binpack=1.0 \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" \
      "$@" | tee "$dir/train.log"
  fi
  notify "Enyo NNUE nonzero-binpack sweep: train done $tag"
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
    --target-clamp 1200 \
    --buckets > "$out"
}

eval_baseline() {
  mkdir -p "$RUN/baseline_eval"
  [ -s "$RUN/baseline_eval/d16.txt" ] \
    || eval_one "$INIT" "$D16_PACKED" 938632 "$RUN/baseline_eval/d16.txt"
  [ -s "$RUN/baseline_eval/lichess.txt" ] \
    || eval_one "$INIT" "$LICHESS_PACKED" 0 "$RUN/baseline_eval/lichess.txt"
  [ -s "$RUN/baseline_eval/selfplay.txt" ] \
    || eval_one "$INIT" "$SELFPLAY_PACKED" 20839426 "$RUN/baseline_eval/selfplay.txt"
  [ -s "$RUN/baseline_eval/binpack_nonzero.txt" ] \
    || eval_one "$INIT" "$BINPACK_PACKED" 1550000 "$RUN/baseline_eval/binpack_nonzero.txt"
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  mkdir -p "$dir/eval"
  notify "Enyo NNUE nonzero-binpack sweep: eval $tag"
  eval_baseline
  eval_one "$dir/model.nn" "$D16_PACKED" 938632 "$dir/eval/d16.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" 0 "$dir/eval/lichess.txt"
  eval_one "$dir/model.nn" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay.txt"
  eval_one "$dir/model.nn" "$BINPACK_PACKED" 1550000 "$dir/eval/binpack_nonzero.txt"

  "$PY" - "$RUN/baseline_eval" "$dir/eval" <<'PY' | tee "$dir/static_summary.txt"
import pathlib
import re
import sys

baseline = pathlib.Path(sys.argv[1])
candidate = pathlib.Path(sys.argv[2])
weights = {
    "d16": 1.25,
    "lichess": 1.0,
    "selfplay": 0.5,
    "binpack_nonzero": 0.35,
}


def metric(path: pathlib.Path, key: str) -> float:
    text = path.read_text()
    m = re.search(rf"^{key}=([-+0-9.]+)%?$", text, re.M)
    if not m:
        raise SystemExit(f"missing {key} in {path}")
    return float(m.group(1))


score = 0.0
max_sign_drop = 0.0
print("dataset          weight base_mae cand_mae delta_mae base_sign cand_sign delta_sign")
for label, weight in weights.items():
    b = baseline / f"{label}.txt"
    c = candidate / f"{label}.txt"
    b_mae = metric(b, "mae")
    c_mae = metric(c, "mae")
    b_sign = metric(b, "sign")
    c_sign = metric(c, "sign")
    d_mae = b_mae - c_mae
    d_sign = c_sign - b_sign
    max_sign_drop = max(max_sign_drop, -d_sign)
    score += weight * d_mae
    print(
        f"{label:16} {weight:6.2f} {b_mae:8.3f} {c_mae:8.3f}"
        f" {d_mae:9.3f} {b_sign:9.2f} {c_sign:9.2f} {d_sign:10.2f}"
    )
print(f"score={score:.3f}")
print(f"max_sign_drop={max_sign_drop:.3f}")
PY
}

static_passes() {
  local summary="$1"
  "$PY" - "$summary" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
score = float(re.search(r"^score=([-+0-9.]+)$", text, re.M).group(1))
max_sign_drop = float(re.search(r"^max_sign_drop=([-+0-9.]+)$", text, re.M).group(1))
rows = {}
for line in text.splitlines():
    parts = line.split()
    if len(parts) == 8 and parts[0] in {"d16", "lichess", "selfplay", "binpack_nonzero"}:
        rows[parts[0]] = {
            "delta_mae": float(parts[4]),
            "delta_sign": float(parts[7]),
        }

ok = (
    score > 5.0
    and max_sign_drop <= 0.35
    and rows.get("d16", {}).get("delta_mae", -999.0) > 0.5
    and rows.get("lichess", {}).get("delta_mae", -999.0) > 0.5
    and rows.get("selfplay", {}).get("delta_sign", -999.0) > -0.35
)
raise SystemExit(0 if ok else 1)
PY
}

run_sprt() {
  local tag="$1"
  local net="$RUN/$tag/model.nn"
  local sprt_dir="$RUN/sprt"
  mkdir -p "$sprt_dir"
  notify "Enyo NNUE nonzero-binpack sweep: SPRT screen start $tag"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --book "$BOOK" \
    --games "$SPRT_GAMES" \
    --tc "$SPRT_TC" \
    --concurrency "$SPRT_CONCURRENCY" \
    --threads "$SPRT_THREADS" \
    --elo0 0 \
    --elo1 "$SPRT_ELO1" \
    --srand 2026051403 \
    --restart "$SPRT_RESTART" \
    --log-dir "$sprt_dir" \
    --name "${tag}_vs_refnet_e5" \
    --candidate-option "Hash=$SPRT_HASH" \
    --candidate-option "nnue_file=$net" \
    --reference-option "Hash=$SPRT_HASH" \
    --reference-option "nnue_file=$INIT" \
    2>&1 | tee "$sprt_dir/${tag}_vs_refnet_e5.log"
  local rc=${PIPESTATUS[0]}
  set -e
  local status
  status=$(rg --no-config "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" \
    "$sprt_dir/${tag}_vs_refnet_e5.log" | tail -6 | tr '\n' ' ')
  notify "Enyo NNUE nonzero-binpack sweep: SPRT finished $tag rc=$rc $status"
  return "$rc"
}

train_candidate huber_nzbin_lr3e7_e8 huber 3e-7
eval_candidate huber_nzbin_lr3e7_e8

train_candidate huber_nzbin_lr1e7_e8 huber 1e-7
eval_candidate huber_nzbin_lr1e7_e8

train_candidate mpe25_nzbin_lr3e7_e8 mpe25 3e-7 --wdl-lambda 1.0
eval_candidate mpe25_nzbin_lr3e7_e8

if [ -n "$ONLY_SPRT_TAG" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: forced SPRT tag $ONLY_SPRT_TAG"
  run_sprt "$ONLY_SPRT_TAG"
  rc=$?
  notify "Enyo NNUE nonzero-binpack sweep: forced SPRT finished rc=$rc run=$RUN"
  exit "$rc"
fi

: > "$RUN/static_passed.tsv"
for tag in huber_nzbin_lr3e7_e8 huber_nzbin_lr1e7_e8 mpe25_nzbin_lr3e7_e8; do
  if static_passes "$RUN/$tag/static_summary.txt"; then
    score=$(awk -F= '/^score=/ {print $2}' "$RUN/$tag/static_summary.txt")
    printf "%s\t%s\n" "$score" "$tag" >> "$RUN/static_passed.tsv"
  else
    notify "Enyo NNUE nonzero-binpack sweep: static reject $tag"
  fi
done

if [ ! -s "$RUN/static_passed.tsv" ]; then
  notify "Enyo NNUE nonzero-binpack sweep: no candidate passed static gate"
  exit 0
fi

best_tag=$(sort -k1,1nr "$RUN/static_passed.tsv" | head -1 | cut -f2)
notify "Enyo NNUE nonzero-binpack sweep: selected $best_tag for SPRT"
run_sprt "$best_tag"
rc=$?
notify "Enyo NNUE nonzero-binpack sweep: finished rc=$rc run=$RUN"
exit "$rc"
