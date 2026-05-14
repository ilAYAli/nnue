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
RUN="${RUN:-$HOME/tmp/enyo_teacher/huber_cp800_neighbors_$(date +%Y%m%d_%H%M%S)}"
INIT="${INIT:-$ENGINE_REPO/nnue/berserk-d43206fe90e4.nn}"
BEST_INIT="${BEST_INIT:-$HOME/tmp/enyo_teacher/objective_sweep_20260513_211559/mix1m_huber_cp800_lr1e6_e8/model.nn}"
ENGINE="${ENGINE:-$HOME/code/cpp/chess/assets/engines/reference}"
SPRT="${SPRT:-$HOME/code/cpp/chess/sprt/sprt}"
BOOK="${BOOK:-$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd}"
PACKED="${PACKED:-$HOME/tmp/enyo_teacher/source_mix_1m_20260513_142124/mix_d16_500k_binpack_500k_all_huber_lr1e6_e8/packed}"

SPRT_GAMES="${SPRT_GAMES:-1000}"
SPRT_CONCURRENCY="${SPRT_CONCURRENCY:-10}"
SPRT_THREADS="${SPRT_THREADS:-2}"
SPRT_HASH="${SPRT_HASH:-512}"
SPRT_TC="${SPRT_TC:-2+0.02}"
SPRT_ELO1="${SPRT_ELO1:-8}"
SPRT_RESTART="${SPRT_RESTART:-off}"
SKIP_TAGS="${SKIP_TAGS:-}"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$NNUE_REPO"

rc=0
trap 'rc=$?; notify "Enyo NNUE huber-cp800 neighbor sweep finished rc=$rc run=$RUN"; exit $rc' EXIT

notify "Enyo NNUE huber-cp800 neighbor sweep start run=$RUN"

for required in "$PY" "$NNUE_REPO" "$ENGINE_REPO" "$TOOLS" "$INIT" \
  "$BEST_INIT" "$ENGINE" "$SPRT" "$BOOK" "$PACKED"; do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE huber-cp800 neighbor sweep missing $required"
    exit 1
  fi
done

run_candidate() {
  local tag="$1"
  local init="$2"
  local lr="$3"
  local epochs="$4"
  local clamp="$5"
  local dir="$RUN/$tag"
  for skip_tag in $SKIP_TAGS; do
    if [ "$skip_tag" = "$tag" ]; then
      notify "Enyo NNUE huber-cp800 neighbor sweep: skip $tag"
      return
    fi
  done
  mkdir -p "$dir"

  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE huber-cp800 neighbor sweep: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$PACKED" \
      --init-from-nn "$init" \
      --objective huber \
      --huber-beta 200 \
      --select-metric mae \
      --epochs "$epochs" \
      --patience 2 \
      --batch-size 8192 \
      --lr "$lr" \
      --weight-decay 1e-6 \
      --target-clamp "$clamp" \
      --device cuda \
      --workers 2 \
      --val-rows 50000 \
      --trainable all \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" | tee "$dir/train.log"
  fi

  mkdir -p "$dir/sprt"
  notify "Enyo NNUE huber-cp800 neighbor sweep: SPRT screen start $tag"
  set +e
  "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "nnue2_file=$dir/model.nn" \
    --reference-option "nnue2_file=$INIT" \
    --candidate-option "Hash=$SPRT_HASH" \
    --reference-option "Hash=$SPRT_HASH" \
    --book "$BOOK" \
    --games "$SPRT_GAMES" \
    --elo0 0 \
    --elo1 "$SPRT_ELO1" \
    --concurrency "$SPRT_CONCURRENCY" \
    --threads "$SPRT_THREADS" \
    --tc "$SPRT_TC" \
    --restart "$SPRT_RESTART" \
    --log-dir "$dir/sprt" \
    --name "${tag}_screen" \
    --eta | tee "$dir/sprt/sprt.log"
  local sprt_rc=${PIPESTATUS[0]}
  set -e

  local final_status
  final_status=$(rg --no-config "^(\\[[ 0-9]+/|Score of|Elo difference:|SPRT:|Finished match|Total Time)" \
    "$dir/sprt/sprt.log" 2>/dev/null | tail -6 | tr "\n" " " || true)
  printf "%s\t%s\t%s\t%s\n" "$tag" "$sprt_rc" "$dir/model.nn" "$final_status" \
    >> "$RUN/results.tsv"
  notify "Enyo NNUE huber-cp800 neighbor sweep: SPRT screen done $tag rc=$sprt_rc ${final_status:-no final status}"
}

: > "$RUN/results.tsv"

# The previous keeper-ish net was this neighborhood:
# Huber(beta=200), target clamp 800, lr=1e-6, 8 epochs, 1M d16/binpack mix.
# Probe small changes around it; do not replay-gate while replay is WIP.
run_candidate "cp800_lr1e6_e8_rerun" "$INIT" 1e-6 8 800
run_candidate "cp800_lr7e7_e8" "$INIT" 7e-7 8 800
run_candidate "cp600_lr1e6_e8" "$INIT" 1e-6 8 600
run_candidate "best_init_cp800_lr2e7_e2" "$BEST_INIT" 2e-7 2 800

notify "Enyo NNUE huber-cp800 neighbor sweep: all screens complete run=$RUN"
