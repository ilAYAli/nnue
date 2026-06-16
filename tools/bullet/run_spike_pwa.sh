#!/usr/bin/env bash
set -euo pipefail

RUN="${RUN:-bullet-spike}"
ROOT="${NNUE_ROOT:-$HOME/code/cpp/chess/nnue}"
SOURCE_JSONL="${SOURCE_JSONL:-$ROOT/runs/imported/fresh_d12self18h64_d16_labels_20260519_113826/score/labeled.jsonl}"
ROWS="${ROWS:-100000}"
MAX_ABS_CP="${MAX_ABS_CP:-1600}"

RUN_DIR="$ROOT/runs/$RUN"
DATA_DIR="$RUN_DIR/data"
OUT_DIR="$RUN_DIR/checkpoints"
TEXT="$DATA_DIR/enyo_${ROWS}.txt"
BULLET_DATA="$DATA_DIR/enyo_${ROWS}.bullet"
LOG="$RUN_DIR/bullet.log"
CURRENT_STAGE="setup"

notify() {
    local event="$1" stage="$2" status="$3" message="${4:-}"
    python3 - "$event" "$stage" "$status" "$message" "$RUN_DIR" "$LOG" <<'PY' \
        | NNUE_RUN_EVENT="$event" "$HOME/scripts/nnue_event_ntfy.sh" || true
import json
import socket
import sys

event, stage, status, message, run, log = sys.argv[1:]
print(json.dumps({
    "event": event,
    "stage": stage,
    "run": run,
    "host": socket.gethostname(),
    "status": status,
    "message": message,
    "log": log,
}))
PY
}

on_error() {
    local rc=$?
    notify fail "$CURRENT_STAGE" fail "Bullet spike failed rc=$rc"
    exit "$rc"
}

trap on_error ERR

mkdir -p "$DATA_DIR" "$OUT_DIR"
cd "$ROOT"

{
    CURRENT_STAGE="bullet_text"
    notify phase_start bullet_text ok "convert JSONL to Bullet text"
    build/fast/nnue-jsonl-to-bullet-text \
        --input "$SOURCE_JSONL" \
        --output "$TEXT" \
        --limit "$ROWS" \
        --max-abs-cp "$MAX_ABS_CP"
    notify phase_done bullet_text ok "Bullet text ready: $TEXT"

    CURRENT_STAGE="bullet_format"
    notify phase_start bullet_format ok "convert Bullet text to BulletFormat"
    source "$HOME/.cargo/env"
    cargo run -q -p bullet-utils --manifest-path "$HOME/source/bullet/Cargo.toml" -- \
        convert --from text --input "$TEXT" --output "$BULLET_DATA"
    cargo run -q -p bullet-utils --manifest-path "$HOME/source/bullet/Cargo.toml" -- \
        validate --input "$BULLET_DATA"
    notify phase_done bullet_format ok "BulletFormat ready: $BULLET_DATA"

    CURRENT_STAGE="bullet_train"
    notify phase_start bullet_train ok "train tiny Reckless-like Bullet net"
    CARGO_TARGET_DIR="$RUN_DIR/cargo-target" \
    CUDA_PATH="${CUDA_PATH:-/usr}" \
    ENYO_BULLET_DATA="$BULLET_DATA" \
    ENYO_BULLET_OUT="$OUT_DIR" \
    ENYO_BULLET_NET_ID="enyo_bullet_spike" \
    ENYO_BULLET_SUPERBATCHES="${SUPERBATCHES:-2048}" \
    ENYO_BULLET_BATCHES="${BATCHES:-64}" \
    ENYO_BULLET_BATCH_SIZE="${BATCH_SIZE:-2048}" \
    ENYO_BULLET_THREADS="${THREADS:-4}" \
    cargo run -q --release --features cuda --manifest-path tools/bullet/spike_trainer/Cargo.toml
    notify done bullet_train ok "Bullet spike complete: $OUT_DIR"
} 2>&1 | tee "$LOG"
