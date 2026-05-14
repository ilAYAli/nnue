#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" "${NOTIFAI_TARGET:-nnue_cmd}" >/dev/null 2>&1 || true
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
RUN="${RUN:-$HOME/tmp/enyo_teacher/signed_bucket_sweep_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"

D16_JSON="${D16_JSON:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl}"
LICHESS_JSON="${LICHESS_JSON:-$HOME/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl}"
BINPACK_JSON="${BINPACK_JSON:-$HOME/tmp/enyo_teacher/nonzero_binpack_sweep_20260514_164049/binpack_nonzero.jsonl}"

D16_PACKED="${D16_PACKED:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed}"
LICHESS_PACKED="${LICHESS_PACKED:-$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k}"
SELFPLAY_PACKED="${SELFPLAY_PACKED:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed}"
BINPACK_PACKED="${BINPACK_PACKED:-$HOME/tmp/enyo_teacher/nonzero_binpack_sweep_20260514_164049/binpack_nonzero_packed}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

notify "Enyo NNUE signed-bucket sweep: start $RUN"

for required in \
  "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS/sample_signed_buckets.py" \
  "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" "$TOOLS/train.py" \
  "$TOOLS/eval_dataset.py" "$INIT" "$ENGINE" "$SPRT" "$BOOK" \
  "$D16_JSON" "$LICHESS_JSON" "$BINPACK_JSON" \
  "$D16_PACKED" "$LICHESS_PACKED" "$SELFPLAY_PACKED" "$BINPACK_PACKED"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE signed-bucket sweep: missing $required"
    exit 1
  fi
done

D16_SAMPLE="$RUN/d16_signed.jsonl"
LICHESS_SAMPLE="$RUN/lichess_signed.jsonl"
BINPACK_SAMPLE="$RUN/binpack_signed.jsonl"
MIXED_JSON="$RUN/source.jsonl"
MIXED_PACKED="$RUN/packed"

if [ ! -s "$D16_SAMPLE" ]; then
  notify "Enyo NNUE signed-bucket sweep: sample d16 signed buckets"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$D16_JSON" \
    --output "$D16_SAMPLE" \
    --seed 2026051404 \
    --progress 250000 \
    --bucket z0050:any:0:50:200000 \
    --bucket p0100:pos:50:100:60000 \
    --bucket n0100:neg:50:100:60000 \
    --bucket p0300:pos:100:300:130000 \
    --bucket n0300:neg:100:300:130000 \
    --bucket p0800:pos:300:800:120000 \
    --bucket n0800:neg:300:800:120000 \
    --bucket p1600:pos:800:1600:7000 \
    --bucket n1600:neg:800:1600:7000 | tee "$RUN/d16_sample.log"
fi

if [ ! -s "$LICHESS_SAMPLE" ]; then
  notify "Enyo NNUE signed-bucket sweep: sample lichess signed buckets"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$LICHESS_JSON" \
    --output "$LICHESS_SAMPLE" \
    --seed 2026051405 \
    --progress 500000 \
    --bucket z0050:any:0:50:350000 \
    --bucket p0100:pos:50:100:160000 \
    --bucket n0100:neg:50:100:160000 \
    --bucket p0300:pos:100:300:400000 \
    --bucket n0300:neg:100:300:400000 \
    --bucket p0800:pos:300:800:450000 \
    --bucket n0800:neg:300:800:450000 \
    --bucket p1600:pos:800:1600:65000 \
    --bucket n1600:neg:800:1600:65000 | tee "$RUN/lichess_sample.log"
fi

if [ ! -s "$BINPACK_SAMPLE" ]; then
  notify "Enyo NNUE signed-bucket sweep: sample binpack signed buckets"
  "$PY" "$TOOLS/sample_signed_buckets.py" \
    --input "$BINPACK_JSON" \
    --output "$BINPACK_SAMPLE" \
    --seed 2026051406 \
    --progress 250000 \
    --bucket p0100:pos:50:100:35000 \
    --bucket n0100:neg:50:100:35000 \
    --bucket p0300:pos:100:300:120000 \
    --bucket n0300:neg:100:300:120000 \
    --bucket p0800:pos:300:800:210000 \
    --bucket n0800:neg:300:800:210000 \
    --bucket p1600:pos:800:1600:220000 \
    --bucket n1600:neg:800:1600:220000 | tee "$RUN/binpack_sample.log"
fi

if [ ! -s "$MIXED_JSON" ]; then
  notify "Enyo NNUE signed-bucket sweep: mix signed bucket rows"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIXED_JSON" \
    --source "$D16_SAMPLE:834000" \
    --source "$LICHESS_SAMPLE:2500000" \
    --source "$BINPACK_SAMPLE:1170000" \
    --seed 2026051407 \
    --progress 500000 | tee "$RUN/mix.log"
fi

if [ ! -s "$MIXED_PACKED/meta.json" ]; then
  notify "Enyo NNUE signed-bucket sweep: pack mixed rows"
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
    notify "Enyo NNUE signed-bucket sweep: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$MIXED_PACKED" \
      --init-from-nn "$INIT" \
      --objective "$objective" \
      --huber-beta 150 \
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
      --source-loss-weight stockfish=1.35 \
      --source-loss-weight lichess_eval=1.0 \
      --source-loss-weight stockfish_binpack=0.55 \
      --source-wdl-lambda stockfish=1.0 \
      --source-wdl-lambda lichess_eval=1.0 \
      --source-wdl-lambda stockfish_binpack=1.0 \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" \
      "$@" | tee "$dir/train.log"
  fi
  notify "Enyo NNUE signed-bucket sweep: train done $tag"
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
  notify "Enyo NNUE signed-bucket sweep: eval $tag"
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
    "d16": 1.35,
    "lichess": 1.0,
    "selfplay": 0.5,
    "binpack_nonzero": 0.55,
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
    score > 4.0
    and max_sign_drop <= 0.20
    and rows.get("d16", {}).get("delta_mae", -999.0) > 0.25
    and rows.get("lichess", {}).get("delta_mae", -999.0) > 0.25
    and rows.get("selfplay", {}).get("delta_sign", -999.0) > -0.25
    and rows.get("binpack_nonzero", {}).get("delta_sign", -999.0) > -0.25
)
raise SystemExit(0 if ok else 1)
PY
}

run_sprt() {
  local tag="$1"
  local net="$RUN/$tag/model.nn"
  local sprt_dir="$RUN/sprt"
  mkdir -p "$sprt_dir"
  notify "Enyo NNUE signed-bucket sweep: 4000-game SPRT start $tag"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --book "$BOOK" \
    --games 4000 \
    --tc 10+0.1 \
    --concurrency 6 \
    --threads 4 \
    --elo0 0 \
    --elo1 5 \
    --srand 2026051408 \
    --log-dir "$sprt_dir" \
    --name "${tag}_vs_refnet_e5" \
    --candidate-option "Hash=1024" \
    --candidate-option "nnue2_file=$net" \
    --reference-option "Hash=1024" \
    --reference-option "nnue2_file=$INIT" \
    2>&1 | tee "$sprt_dir/${tag}_vs_refnet_e5.log"
  local rc=${PIPESTATUS[0]}
  set -e
  local status
  status=$(rg "Elo difference|SPRT:|Finished match|Total Time|^\\[[[:space:][:digit:]]+/" \
    "$sprt_dir/${tag}_vs_refnet_e5.log" | tail -6 | tr '\n' ' ')
  notify "Enyo NNUE signed-bucket sweep: SPRT finished $tag rc=$rc $status"
  return "$rc"
}

train_candidate signed_mpe25_lr3e7_e8 mpe25 3e-7 --wdl-lambda 1.0
eval_candidate signed_mpe25_lr3e7_e8

train_candidate signed_huber_lr3e7_e8 huber 3e-7
eval_candidate signed_huber_lr3e7_e8

train_candidate signed_mpe25_lr1e7_e8 mpe25 1e-7 --wdl-lambda 1.0
eval_candidate signed_mpe25_lr1e7_e8

: > "$RUN/static_passed.tsv"
for tag in signed_mpe25_lr3e7_e8 signed_huber_lr3e7_e8 signed_mpe25_lr1e7_e8; do
  if static_passes "$RUN/$tag/static_summary.txt"; then
    score=$(awk -F= '/^score=/ {print $2}' "$RUN/$tag/static_summary.txt")
    printf "%s\t%s\n" "$score" "$tag" >> "$RUN/static_passed.tsv"
  else
    notify "Enyo NNUE signed-bucket sweep: static reject $tag"
  fi
done

if [ ! -s "$RUN/static_passed.tsv" ]; then
  notify "Enyo NNUE signed-bucket sweep: no candidate passed static gate"
  exit 0
fi

best_tag=$(sort -k1,1nr "$RUN/static_passed.tsv" | head -1 | cut -f2)
notify "Enyo NNUE signed-bucket sweep: selected $best_tag for SPRT"
run_sprt "$best_tag"
rc=$?
notify "Enyo NNUE signed-bucket sweep: finished rc=$rc run=$RUN"
exit "$rc"
