#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" "${CODEX_TMUX:-nnue_test}" >/dev/null 2>&1 || true
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
TOOLS="$NNUE_REPO/tools/nnue2"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MIX="${MIX:-$HOME/tmp/enyo_teacher/source_mix_cp800_5m_$STAMP}"
RUN="${RUN:-$HOME/tmp/enyo_teacher/objective_sweep_cp800_5m_$STAMP}"

D16_JSON="${D16_JSON:-$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl}"
SELFPLAY_JSON="${SELFPLAY_JSON:-$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl}"
BINPACK_JSON="${BINPACK_JSON:-$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/binpack.jsonl}"

mkdir -p "$MIX"
exec > >(tee -a "$MIX/run.log") 2>&1
cd "$NNUE_REPO"

notify "Enyo NNUE cp800 5M sweep: start mix=$MIX run=$RUN"

for required in \
  "$PY" "$NNUE_REPO" "$TOOLS/mix_jsonl.py" "$TOOLS/pack_dataset.py" \
  "$TOOLS/run_objective_sweep_pwa.sh" "$D16_JSON" "$SELFPLAY_JSON" "$BINPACK_JSON"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE cp800 5M sweep: missing $required"
    exit 1
  fi
done

if [ ! -s "$MIX/source.jsonl" ]; then
  notify "Enyo NNUE cp800 5M sweep: mix 0.9M d16 + 2.5M selfplay + 1.6M binpack"
  "$PY" "$TOOLS/mix_jsonl.py" \
    --output "$MIX/source.jsonl" \
    --source "$D16_JSON:900000" \
    --source "$SELFPLAY_JSON:2500000" \
    --source "$BINPACK_JSON:1600000" \
    --seed 2026051401 \
    --progress 500000 | tee "$MIX/mix.log"
fi

if [ ! -s "$MIX/packed/meta.json" ]; then
  notify "Enyo NNUE cp800 5M sweep: pack mixed rows"
  "$PY" "$TOOLS/pack_dataset.py" \
    --input "$MIX/source.jsonl" \
    --out-dir "$MIX/packed" \
    --progress 500000 | tee "$MIX/pack.log"
fi

notify "Enyo NNUE cp800 5M sweep: launch objective gate run"
RUN="$RUN" MIXED_PACKED="$MIX/packed" "$TOOLS/run_objective_sweep_pwa.sh"
rc=$?
notify "Enyo NNUE cp800 5M sweep: finished rc=$rc run=$RUN"
exit "$rc"
