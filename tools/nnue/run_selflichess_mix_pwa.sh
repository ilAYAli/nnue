#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" >/dev/null 2>&1 || true
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
RUN="${RUN:-$HOME/tmp/enyo_teacher/selflichess_mix_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"
SELFPLAY_JSON="${SELFPLAY_JSON:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl}"
LICHESS_JSON="${LICHESS_JSON:-$HOME/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl}"

SELFPLAY_ROWS="${SELFPLAY_ROWS:-8000000}"
LICHESS_ROWS="${LICHESS_ROWS:-2000000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
TRAIN_LR="${TRAIN_LR:-3e-7}"
TARGET_CLAMP="${TARGET_CLAMP:-1200}"
SPRT_GAMES="${SPRT_GAMES:-4000}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_ELO1="${SPRT_ELO1:-8}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE selflichess mix finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE selflichess mix start run=$RUN"

for required in "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS" "$INIT" "$ENGINE" \
  "$SPRT" "$BOOK" "$SELFPLAY_JSON" "$LICHESS_JSON"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE selflichess mix missing $required"
    exit 1
  fi
done

MIX="$RUN/source_self${SELFPLAY_ROWS}_lichess${LICHESS_ROWS}.jsonl"
PACKED="$RUN/packed"
TAG="self${SELFPLAY_ROWS}_lichess${LICHESS_ROWS}_mpe_lr${TRAIN_LR}_e${TRAIN_EPOCHS}_cp${TARGET_CLAMP}"
CAND_DIR="$RUN/$TAG"
mkdir -p "$CAND_DIR"

if [ ! -s "$MIX" ]; then
  notify "Enyo NNUE selflichess mix: create source rows"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIX" \
    --source "$SELFPLAY_JSON:$SELFPLAY_ROWS" \
    --source "$LICHESS_JSON:$LICHESS_ROWS" \
    --seed 2026051501 \
    --progress 500000
fi

if [ ! -s "$PACKED/meta.json" ]; then
  notify "Enyo NNUE selflichess mix: pack rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$MIX" \
    --out-dir "$PACKED" \
    --progress 500000
fi

if [ ! -s "$CAND_DIR/model.nn" ]; then
  notify "Enyo NNUE selflichess mix: train $TAG"
  "$PY" "$TOOLS/train.py" \
    --data "$PACKED" \
    --init-from-nn "$INIT" \
    --objective mpe25 \
    --wdl-lambda 0.95 \
    --source-wdl-lambda stockfish=0.95 \
    --source-wdl-lambda lichess_eval=1.0 \
    --select-metric loss \
    --epochs "$TRAIN_EPOCHS" \
    --patience 2 \
    --batch-size 8192 \
    --lr "$TRAIN_LR" \
    --weight-decay 1e-6 \
    --target-clamp "$TARGET_CLAMP" \
    --device cuda \
    --workers 2 \
    --val-rows 100000 \
    --trainable all \
    --out "$CAND_DIR/model.pt" \
    --out-nn "$CAND_DIR/model.nn" | tee "$CAND_DIR/train.log"
fi

notify "Enyo NNUE selflichess mix: static eval $TAG"
"$PY" "$TOOLS/eval_dataset.py" \
  --net "$INIT" \
  --data "$PACKED" \
  --skip 9900000 \
  --rows 100000 \
  --batch-size 8192 \
  --device cuda \
  --target-clamp "$TARGET_CLAMP" \
  --sources > "$CAND_DIR/baseline_eval.txt"
"$PY" "$TOOLS/eval_dataset.py" \
  --net "$CAND_DIR/model.nn" \
  --data "$PACKED" \
  --skip 9900000 \
  --rows 100000 \
  --batch-size 8192 \
  --device cuda \
  --target-clamp "$TARGET_CLAMP" \
  --sources > "$CAND_DIR/candidate_eval.txt"
{
  echo "-- baseline --"
  cat "$CAND_DIR/baseline_eval.txt"
  echo "-- candidate --"
  cat "$CAND_DIR/candidate_eval.txt"
} | tee "$CAND_DIR/static_summary.txt"

GATE="$CAND_DIR/sprt"
mkdir -p "$GATE"

notify "Enyo NNUE selflichess mix: 4000-game SPRT start $TAG"
"$SPRT" \
  --candidate "$ENGINE" \
  --reference "$ENGINE" \
  --candidate-option "nnue_file=$CAND_DIR/model.nn" \
  --reference-option "nnue_file=$INIT" \
  --candidate-option "Hash=$SPRT_HASH" \
  --reference-option "Hash=$SPRT_HASH" \
  --book "$BOOK" \
  --games "$SPRT_GAMES" \
  --concurrency "$SPRT_CONCURRENCY" \
  --threads "$SPRT_THREADS" \
  --tc "$SPRT_TC" \
  --elo1 "$SPRT_ELO1" \
  --log-dir "$GATE" \
  --name "$TAG" \
  --eta | tee "$GATE/sprt.log"

notify "Enyo NNUE selflichess mix: SPRT done $TAG"
