#!/usr/bin/env bash
set -euo pipefail

payload=${NNUE_RUN_EVENT_JSON:-}
if [ -z "$payload" ]; then
    payload=$(cat)
fi

NNUE_RUN_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
from pathlib import Path

event = json.loads(os.environ["NNUE_RUN_EVENT_PAYLOAD"])
name = event.get("name") or Path(str(event.get("run", ""))).name or "nnue-run"
kind = event.get("event", "event")
stage = event.get("stage", "")
status = event.get("status", "")
message = str(event.get("message") or "").strip()
log = event.get("log", "")

print(f"[{kind}] {name}" + (f" {stage}" if stage else "") + (f" {status}" if status else ""))
if message:
    print(message)
if log:
    print(f"log: {log}")
PY
