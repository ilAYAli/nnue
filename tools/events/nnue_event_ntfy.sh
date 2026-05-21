#!/usr/bin/env bash
set -euo pipefail

# Generic NNUE event hook. It reads a JSON event from NNUE_RUN_EVENT_JSON or
# stdin, publishes a human status update, and nudges AI_stdin when a phase needs
# follow-up attention.

NNUE_URL=${NNUE_NTFY_URL:-https://ntfy.wahlman.no/nnue}
AI_STDIN_URL=${NNUE_AI_STDIN_URL:-https://ntfy.wahlman.no/AI_stdin}
DRY_RUN=${NNUE_NTFY_DRY_RUN:-0}
AI_STDIN_DRY_RUN=${NNUE_AI_STDIN_DRY_RUN:-$DRY_RUN}
AI_STDIN_EVENTS=${NNUE_AI_STDIN_EVENTS:-done,fail}
LOG=${NNUE_NTFY_LOG:-$HOME/tmp/nnue_event_ntfy.log}

if [ "${1:-}" = "test" ]; then
    shift || true
    payload=$(python3 - <<'PY'
import json
import socket

print(json.dumps({
    "event": "test",
    "stage": "manual",
    "run": "nnue_event_ntfy_test",
    "host": socket.gethostname(),
    "status": "ok",
    "message": "manual nnue_event_ntfy.sh test",
}))
PY
)
    export NNUE_RUN_EVENT="test"
    export NNUE_RUN_EVENT_JSON="$payload"
fi

payload=${NNUE_RUN_EVENT_JSON:-}
if [ -z "$payload" ]; then
    payload=$(cat)
fi

source "$HOME/.ntfy" 2>/dev/null || true
mkdir -p "$(dirname "$LOG")"

rendered=$(
    NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
from pathlib import Path

event = json.loads(os.environ["NNUE_EVENT_PAYLOAD"])
name = Path(event.get("run", "")).name or event.get("run", "")
event_name = event.get("event", "unknown")
stage = event.get("stage", "")
status = event.get("status", "")
log = event.get("log", "")

lines = ["Current task"]
lines.append(f"  • Event: {event_name}")
if stage:
    lines.append(f"  • Stage: {stage}")
if name:
    lines.append(f"  • Run: {name}")
if event.get("host"):
    lines.append(f"  • Host: {event['host']}")
if status:
    lines.append(f"  • Status: {status}")
if "rc" in event:
    lines.append(f"  • RC: {event['rc']}")
if event.get("message"):
    lines.append(f"  • Message: {event['message']}")
if log:
    lines.append(f"  • Log: {log}")
if event.get("tag"):
    lines.append(f"  • Candidate: {event['tag']}")
if event.get("candidate_net"):
    lines.append(f"  • Candidate net: {event['candidate_net']}")
if event.get("games"):
    lines.append(f"  • Games: {event['games']}")
if event.get("tc"):
    lines.append(f"  • Time control: {event['tc']}")

if event_name == "done" and isinstance(event.get("validation_commands"), list):
    commands = event["validation_commands"]
    if commands:
        lines.append("")
        lines.append("Validation commands")
        lines.append("```sh")
        lines.extend(str(command) for command in commands)
        lines.append("```")
elif event_name == "done":
    lines.append("")
    lines.append("Next")
    lines.append("  • Next: inspect result and start the next task")
else:
    lines.append("")
    lines.append("ETA")
    if event_name == "fail":
        lines.append("  • Next: inspect log and fix failed phase")
    elif event_name == "phase_done":
        lines.append("  • Next: next configured phase starts immediately")
    elif event_name == "phase_start":
        lines.append("  • Phase completion: reported immediately by this hook")
    else:
        lines.append("  • Next update: next phase transition")

if event_name == "fail":
    prompt = f"NNUE phase failed: run={name} stage={stage or 'n/a'} status={status or 'failed'} log={log}. Inspect the log and fix the failed phase."
elif event_name == "done":
    prompt = f"NNUE run complete: run={name} status={status or 'ok'} log={log}. Inspect the result and start the next task."
elif event_name == "phase_done":
    prompt = f"NNUE stage complete, start next task: run={name} stage={stage or 'n/a'} status={status or 'ok'} log={log}."
else:
    prompt = ""

print("\n".join(lines))
print("__NNUE_AI_PROMPT__" + prompt)
PY
)

ai_prompt=$(printf '%s\n' "$rendered" | sed -n 's/^__NNUE_AI_PROMPT__//p' | tail -1)
body=$(printf '%s\n' "$rendered" | sed '/^__NNUE_AI_PROMPT__/d')

publish() {
    local url="$1"
    local data="$2"
    local title="$3"
    local priority="$4"

    if [ -n "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" ]; then
        curl -fsS -m 10 \
            -u "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" \
            -H "Title: $title" \
            -H "Priority: $priority" \
            -H "X-Markdown: yes" \
            -w "%{http_code}" \
            -o /dev/null \
            --data-binary "$data" \
            "$url"
    else
        curl -fsS -m 10 \
            -H "Title: $title" \
            -H "Priority: $priority" \
            -H "X-Markdown: yes" \
            -w "%{http_code}" \
            -o /dev/null \
            --data-binary "$data" \
            "$url"
    fi
}

event_name=${NNUE_RUN_EVENT:-}
if [ -z "$event_name" ]; then
    event_name=$(NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["NNUE_EVENT_PAYLOAD"]).get("event", "event"))
PY
)
fi
priority=3
case "$event_name" in
    fail)
        priority=5
        ;;
    done|phase_done)
        priority=4
        ;;
esac

if [ "$DRY_RUN" = "1" ]; then
    printf '%s\n' "$body"
    nnue_http="dry-run"
else
    nnue_http=$(publish "$NNUE_URL" "$body" "Enyo NNUE $event_name" "$priority")
fi

ai_http="skip"
case ",$AI_STDIN_EVENTS," in
    *,"$event_name",*)
        if [ -n "$ai_prompt" ]; then
            if [ "$AI_STDIN_DRY_RUN" = "1" ]; then
                printf '\nAI_stdin\n  • %s\n' "$ai_prompt"
                ai_http="dry-run"
            else
                ai_http=$(publish "$AI_STDIN_URL" "$ai_prompt" "Enyo NNUE $event_name" "$priority")
            fi
        fi
        ;;
esac

printf '%s event=%s nnue=%s ai_stdin=%s nnue_http=%s ai_http=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$NNUE_URL" "$AI_STDIN_URL" "$nnue_http" "$ai_http" >>"$LOG"
