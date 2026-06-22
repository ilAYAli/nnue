#!/usr/bin/env bash
set -euo pipefail

# NNUE event hook.
#
# Routing contract:
# - AI_stdin wakes the agent and is controlled by NNUE_AI_STDIN_EVENTS.
# - AI_stdout gets concise done/fail/test summaries for user-visible progress.
# - The nnue topic is filtered to good-news/conclusion events unless
#   NNUE_USER_NOTIFY_GENERIC=1.

NNUE_URL=${NNUE_NTFY_URL:-https://ntfy.wahlman.no/nnue}
AI_STDIN_URL=${NNUE_AI_STDIN_URL:-https://ntfy.wahlman.no/AI_stdin}
AI_STDOUT_URL=${NNUE_AI_STDOUT_URL:-https://ntfy.wahlman.no/AI_stdout}
EVENTS=${NNUE_NTFY_EVENTS-done,fail,test}
AI_EVENTS=${NNUE_AI_STDIN_EVENTS-phase_done,done,fail}
AI_STDOUT_EVENTS=${NNUE_AI_STDOUT_EVENTS-done,fail}
USER_GENERIC=${NNUE_USER_NOTIFY_GENERIC:-0}
AI_ENABLE=${NNUE_AI_STDIN_ENABLE:-1}
AI_STDIN_NTFY_ENABLE=${NNUE_AI_STDIN_NTFY_ENABLE:-1}
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

user_worthy=$(NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY'
import json
import os
import re
from pathlib import Path

event = json.loads(os.environ["NNUE_EVENT_PAYLOAD"])
name = str(event.get("event", ""))
if name == "test":
    print("1")
elif any(event.get(key) for key in (
        "user_notify", "good_news", "promotion_candidate", "improved",
        "critical", "critical_failure")):
    print("1")
elif name == "done":
    message = str(event.get("message") or "")
    log = event.get("log", "")
    try:
        path = Path(str(log)).expanduser()
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - 700_000))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        text = ""
    text = text + "\n" + message
    sprt_lines = [
        line.strip()
        for line in text.splitlines()
        if (
            "Enyo NNUE SPRT finished" in line
            or re.match(r"^\[\s*\d+/\d+\]", line)
            or re.search(r"\bElo(?: difference|:)?\s+[+-]?\d", line)
        )
    ]
    worthy = False
    if sprt_lines:
        line = sprt_lines[-1]
        elo_match = re.search(r"Elo(?: difference|:)?\s+([+-]?\d+(?:\.\d+)?)", line)
        llr_match = re.search(r"LLR\s+([+-]?\d+(?:\.\d+)?)/", line)
        los_match = re.search(r"LOS\s+([0-9.]+)%", line)
        if elo_match:
            elo = float(elo_match.group(1))
            llr = float(llr_match.group(1)) if llr_match else 0.0
            los = float(los_match.group(1)) if los_match else 0.0
            worthy = elo > 0.0 and (los >= 80.0 or llr > 0.0)
    print("1" if worthy else "0")
else:
    print("0")
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
if [ "$USER_GENERIC" != "1" ] && [ "$user_worthy" != "1" ]; then
    send_nnue=0
fi

if [ "$send_nnue" = "0" ] && [ "$send_ai_stdout" = "0" ] && [ "$send_ai_stdin" = "0" ]; then
    printf '%s event=%s skipped\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
    exit 0
fi

if [ -r "$HOME/.ntfy" ]; then
    source "$HOME/.ntfy"
fi

rendered=$(
    NNUE_EVENT_PAYLOAD="$payload" python3 - <<'PY_RENDER'
import json
import os
import re
from pathlib import Path


event = json.loads(os.environ["NNUE_EVENT_PAYLOAD"])
run_dir = Path(str(event.get("run", ""))).expanduser()
name = run_dir.name or str(event.get("run", "")) or "unknown"
event_name = str(event.get("event", "unknown"))
stage = str(event.get("stage", "") or "")
status = str(event.get("status", "") or "")
log = str(event.get("log", "") or "")
message = str(event.get("message", "") or "")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def tail_text(path, size=700_000):
    if not path:
        return ""
    p = Path(path).expanduser()
    try:
        with p.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def hypothesis():
    direct = str(event.get("hypothesis") or "").strip()
    if direct:
        return direct
    for candidate in (
        run_dir / "build.resolved.json",
        run_dir / "build.json",
        Path("build.json"),
    ):
        data = read_json(candidate)
        if not data:
            continue
        if candidate == Path("build.json") and str(data.get("run") or "") != name:
            continue
        value = str(data.get("hypothesis") or "").strip()
        if value:
            return value
    return ""


def final_status():
    haystack = "\n".join(part for part in (message, tail_text(log)) if part)
    for line in reversed(haystack.splitlines()):
        stripped = line.strip()
        if "tasks=" in stripped and "games=" in stripped and "elo=" in stripped:
            return stripped[stripped.find("tasks="):]
        if stripped.startswith("Crucible ") and " elo=" in stripped:
            return stripped
    return ""


def failure_error():
    text = tail_text(log)
    patterns = (
        r"^FAILED:\s*(.+)$",
        r"^error:\s*(.+)$",
        r"^thread .+ panicked .*$",
        r"^.*No space left on device.*$",
        r"^.*CUDA_ERROR.*$",
        r"^.*Traceback.*$",
        r"^.*ValueError:.*$",
    )
    for pattern in patterns:
        matches = [line.strip() for line in text.splitlines() if re.search(pattern, line)]
        if matches:
            return matches[-1]
    if message:
        return message.strip()
    return status or "failed"


title = f"NNUE {event_name}"
lines = [title, f"  • Run: {name}"]
hyp = hypothesis()
if hyp:
    lines.append(f"  • Hypothesis: {hyp}")
if event_name == "fail":
    lines.append(f"  • Error: {failure_error()}")
else:
    lines.append(f"  • Status: {final_status() or status or event_name}")

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
PY_RENDER
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

is_worker_target() {
    case "$1" in
        nnue_native|nnue_native:*|nnue_reckless|nnue_reckless:*|\
        nnue_training|nnue_training:*|nnue_test|nnue_test:*)
            return 0
            ;;
    esac
    return 1
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
    local rc=0
    local notifai_delivered=0

    if [ "$DRY_RUN" = "1" ]; then
        printf '%s\n' "$prompt"
        return
    fi

    if is_worker_target "$NOTIFAI_TARGET"; then
        printf '%s event=%s notifai_refused_worker target=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" \
            "$NOTIFAI_TARGET" >>"$LOG"
    elif [ -x "$NOTIFAI" ]; then
        if "$NOTIFAI" "$prompt" "$NOTIFAI_TARGET" >/dev/null 2>&1; then
            notifai_delivered=1
            printf '%s event=%s notifai_ok target=%s\n' \
                "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" \
                "$NOTIFAI_TARGET" >>"$LOG"
        else
            rc=$?
            printf '%s event=%s notifai_failed rc=%s target=%s\n' \
                "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$rc" \
                "$NOTIFAI_TARGET" >>"$LOG"
        fi
    else
        printf '%s event=%s notifai_missing path=%s target=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$NOTIFAI" \
            "$NOTIFAI_TARGET" >>"$LOG"
    fi

    if [ "$AI_STDIN_NTFY_ENABLE" != "1" ]; then
        printf '%s event=%s ai_stdin_ntfy_skipped\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
        return 0
    fi

    if [ "$notifai_delivered" = "1" ]; then
        printf '%s event=%s ai_stdin_ntfy_skipped_notifai_ok\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
        return 0
    fi

    if publish "$AI_STDIN_URL" "$prompt" "$title" "$priority"; then
        printf '%s event=%s ai_stdin_ntfy_ok\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
    else
        rc=$?
        printf '%s event=%s ai_stdin_ntfy_failed rc=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$rc" \
            >>"$LOG"
        return "$rc"
    fi
}

if [ "$send_ai_stdin" = "1" ] && [ -n "$ai_prompt" ]; then
    publish_ai "$ai_prompt" "Enyo NNUE $event_name" "$priority"
    printf '%s event=%s ai_wakeup_processed\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
fi

printf '%s event=%s sent\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
