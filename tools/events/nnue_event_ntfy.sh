#!/usr/bin/env bash
set -euo pipefail

# NNUE event hook. It publishes only completion/failure by default to avoid
# phase spam. Set NNUE_NTFY_EVENTS=phase_done,done,fail for verbose runs.

NNUE_URL=${NNUE_NTFY_URL:-https://ntfy.wahlman.no/nnue}
AI_STDIN_URL=${NNUE_AI_STDIN_URL:-https://ntfy.wahlman.no/AI_stdin}
AI_STDOUT_URL=${NNUE_AI_STDOUT_URL:-https://ntfy.wahlman.no/AI_stdout}
EVENTS=${NNUE_NTFY_EVENTS:-done,fail,test}
AI_EVENTS=${NNUE_AI_STDIN_EVENTS:-done,fail}
AI_STDOUT_EVENTS=${NNUE_AI_STDOUT_EVENTS:-done,fail}
AI_ENABLE=${NNUE_AI_STDIN_ENABLE:-1}
AI_STDOUT_ENABLE=${NNUE_AI_STDOUT_ENABLE:-1}
NOTIFAI=${NNUE_NOTIFAI:-$HOME/scripts/notifai.sh}
NOTIFAI_TARGET=${NNUE_NOTIFAI_TARGET:-${NOTIFAI_TARGET:-codex_1}}
DRY_RUN=${NNUE_NTFY_DRY_RUN:-0}
LOG=${NNUE_NTFY_LOG:-$HOME/tmp/nnue_event_ntfy.log}

payload=${NNUE_RUN_EVENT_JSON:-}
if [ -z "$payload" ]; then
    payload=$(cat)
fi

event_name=$(NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["NNUE_EVENT_PAYLOAD"]).get("event", "event"))
PY
)

mkdir -p "$(dirname "$LOG")"
send_nnue=0
send_ai_stdout=0
send_ai_stdin=0

case ",$EVENTS," in
    *,"$event_name",*) send_nnue=1 ;;
esac
if [ "$AI_STDOUT_ENABLE" = "1" ]; then
    case ",$AI_STDOUT_EVENTS," in
        *,"$event_name",*) send_ai_stdout=1 ;;
    esac
fi
if [ "$AI_ENABLE" = "1" ]; then
    case ",$AI_EVENTS," in
        *,"$event_name",*) send_ai_stdin=1 ;;
    esac
fi

if [ "$send_nnue" = "0" ] && [ "$send_ai_stdout" = "0" ] && [ "$send_ai_stdin" = "0" ]; then
    printf '%s event=%s skipped\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
    exit 0
fi

source "$HOME/.ntfy" 2>/dev/null || true

rendered=$(
    NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
from pathlib import Path

event = json.loads(os.environ["NNUE_EVENT_PAYLOAD"])
name = Path(event.get("run", "")).name or str(event.get("run", ""))
event_name = event.get("event", "unknown")
stage = event.get("stage", "")
status = event.get("status", "")
log = event.get("log", "")

lines = ["Current task"]
lines.append(f"  • Task: {event_name}")
if stage:
    lines.append(f"  • Stage: {stage}")
if name:
    lines.append(f"  • Run: {name}")
if event.get("host"):
    lines.append(f"  • Host: {event['host']}")
if status:
    lines.append(f"  • State: {status}")
if "rc" in event:
    lines.append(f"  • RC: {event['rc']}")
if log:
    lines.append(f"  • Log: {log}")
if event.get("candidate_net"):
    lines.append(f"  • Candidate net: {event['candidate_net']}")

lines.append("")
lines.append("ETA")
if event_name == "fail":
    lines.append("  • Next: inspect log and fix the failed phase")
elif event_name == "done":
    lines.append("  • Next: inspect result and choose the next gate")
elif event_name == "phase_done":
    lines.append("  • Next: inspect result and continue the next task")
else:
    lines.append("  • Next: no action unless this was unexpected")

if event_name == "fail":
    prompt = f"NNUE phase failed: run={name} stage={stage or 'n/a'} status={status or 'failed'} log={log}. Inspect the log and fix the failed phase."
elif event_name == "done":
    prompt = f"NNUE run complete: run={name} status={status or 'ok'} log={log}. Inspect the result and start the next task."
elif event_name == "phase_done":
    prompt = f"NNUE phase complete: run={name} stage={stage or 'n/a'} status={status or 'ok'} log={log}. Inspect the result and continue the next task."
else:
    prompt = ""

print("\n".join(lines))
print("__AI_PROMPT__" + prompt)
PY
)

body=$(printf '%s\n' "$rendered" | sed '/^__AI_PROMPT__/d')
ai_prompt=$(printf '%s\n' "$rendered" | sed -n 's/^__AI_PROMPT__//p' | tail -1)
stdout_body=$(printf '<output>\n%s\n</output>\n' "$body")
if [ -n "$ai_prompt" ]; then
    stdout_body=$(printf '%s\n\n<summary>\n%s\n</summary>\n' "$stdout_body" "$ai_prompt")
fi

publish() {
    local url="$1"
    local data="$2"
    local title="$3"
    local priority="$4"

    if [ "$DRY_RUN" = "1" ]; then
        printf '%s\n' "$data"
        return
    fi

    if [ -n "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" ]; then
        curl -fsS -m 10 -u "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" \
            -H "Title: $title" -H "Priority: $priority" \
            --data-binary "$data" "$url" >/dev/null
    else
        curl -fsS -m 10 \
            -H "Title: $title" -H "Priority: $priority" \
            --data-binary "$data" "$url" >/dev/null
    fi
}

priority=3
case "$event_name" in
    fail) priority=5 ;;
    done) priority=4 ;;
esac

if [ "$send_nnue" = "1" ]; then
    publish "$NNUE_URL" "$body" "Enyo NNUE $event_name" "$priority"
    printf '%s event=%s nnue_sent\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
fi

if [ "$send_ai_stdout" = "1" ]; then
    publish "$AI_STDOUT_URL" "$stdout_body" "Enyo NNUE $event_name" "$priority"
    printf '%s event=%s ai_stdout_sent\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
fi

publish_ai() {
    local prompt="$1"
    local title="$2"
    local priority="$3"

    if [ "$DRY_RUN" = "1" ]; then
        printf '%s\n' "$prompt"
        return
    fi

    if [ -x "$NOTIFAI" ]; then
        "$NOTIFAI" "$prompt" "$NOTIFAI_TARGET" >/dev/null 2>&1 || true
    fi
    publish "$AI_STDIN_URL" "$prompt" "$title" "$priority"
}

if [ "$send_ai_stdin" = "1" ] && [ -n "$ai_prompt" ]; then
    publish_ai "$ai_prompt" "Enyo NNUE $event_name" "$priority"
    printf '%s event=%s ai_stdin_sent\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
fi

printf '%s event=%s sent\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
