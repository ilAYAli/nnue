#!/usr/bin/env bash
set -euo pipefail

# NNUE event hook.
#
# Routing:
#   fail            → ping
#   iteration_done  → nnue (phone notification)
#   everything else → AI_stdout
#   AI_stdin        → opt-in via NNUE_AI_STDIN_EVENTS

NNUE_URL=${NNUE_NTFY_URL:-https://ntfy.wahlman.no/nnue}
AI_STDIN_URL=${NNUE_AI_STDIN_URL:-https://ntfy.wahlman.no/AI_stdin}
AI_STDOUT_URL=${NNUE_AI_STDOUT_URL:-https://ntfy.wahlman.no/AI_stdout}
PING_URL=${NNUE_PING_URL:-https://ntfy.wahlman.no/ping}
AI_STDIN_EVENTS=${NNUE_AI_STDIN_EVENTS:-}
AI_STDIN_ENABLE=${NNUE_AI_STDIN_ENABLE:-1}
NOTIFAI_ENABLE=${NNUE_NOTIFAI_ENABLE:-1}
NOTIFAI_COMMAND=${NNUE_NOTIFAI_COMMAND:-notifai.sh}
NOTIFAI_TARGET=${NNUE_NOTIFAI_TARGET:-AI_stdin:1.1}
DRY_RUN=${NNUE_NTFY_DRY_RUN:-0}
LOG=${NNUE_NTFY_LOG:-$HOME/tmp/nnue_event_ntfy.log}

payload=${NNUE_RUN_EVENT_JSON:-}
if [ -z "$payload" ]; then
    payload=$(cat)
fi

event_name=$(printf '%s' "$payload" | python3 -c "import json,sys; print(json.load(sys.stdin).get('event',''))")

mkdir -p "$(dirname "$LOG")"

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
        if stripped.startswith("Forge ") and " elo=" in stripped:
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


def first_match(patterns, text):
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return found.group(1).strip()
    return ""


def percent(value):
    if not value:
        return ""
    return value if value.endswith("%") else f"{value}%"


def parse_sprt_line(line):
    metrics = {}
    metrics["games"] = first_match((
        r"\bgames=([0-9]+/[0-9]+)",
        r"^\[\s*([0-9]+/[0-9]+)\]",
    ), line)
    metrics["tasks"] = first_match((r"\btasks=([0-9]+/[0-9]+)",), line)
    metrics["elo"] = first_match((
        r"\belo=([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"\bElo\s+([+-]?[0-9]+(?:\.[0-9]+)?)",
    ), line)
    metrics["ci"] = first_match((
        r"\bci=([0-9]+(?:\.[0-9]+)?)",
        r"\+/-\s*([0-9]+(?:\.[0-9]+)?)",
    ), line)
    metrics["llr"] = first_match((
        r"\bllr=([+-]?[0-9]+(?:\.[0-9]+)?/[+-]?[0-9]+(?:\.[0-9]+)?(?:\s*\([^)]+\))?)",
        r"\bLLR\s+([+-]?[0-9]+(?:\.[0-9]+)?/[+-]?[0-9]+(?:\.[0-9]+)?(?:\s*\([^)]+\))?)",
    ), line)
    metrics["los"] = percent(first_match((
        r"\blos=([0-9]+(?:\.[0-9]+)?%?)",
        r"\bLOS\s+([0-9]+(?:\.[0-9]+)?%?)",
    ), line))
    metrics["draw"] = percent(first_match((
        r"\bdraw=([0-9]+(?:\.[0-9]+)?%?)",
        r"\bdraw\s+([0-9]+(?:\.[0-9]+)?%?)",
    ), line))
    return {key: value for key, value in metrics.items() if value}


def sprt_metrics():
    sprt = event.get("sprt") if isinstance(event.get("sprt"), dict) else {}
    sources = [
        str(sprt.get("line") or ""),
        message,
        status,
        tail_text(log),
    ]
    for source in sources:
        for line in reversed(str(source).splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            if "games=" in stripped or re.match(r"^\[\s*[0-9]+/[0-9]+\]", stripped):
                parsed = parse_sprt_line(stripped)
                if parsed:
                    break
        else:
            continue
        if parsed:
            break
    else:
        parsed = {}

    for key in ("elo", "llr"):
        value = str(sprt.get(key) or "").strip()
        if value and key not in parsed:
            parsed[key] = value
    return parsed


def sprt_status_text():
    lowered = message.lower()
    if "passed" in lowered:
        return "passed"
    if "inconclusive" in lowered:
        return "inconclusive"
    if "rejected" in lowered or "failed" in lowered:
        return "failed"
    return status or event_name


is_sprt = bool(event.get("sprt")) or stage.startswith("sprt") or "SPRT" in message
title = "NNUE SPRT" if is_sprt else f"NNUE {event_name}"
lines = [title, f"  • Run: {name}"]
if stage:
    lines.append(f"  • Stage: {stage}")
hyp = hypothesis()
if hyp:
    lines.append(f"  • Hypothesis: {hyp}")
if event_name == "fail":
    lines.append(f"  • Error: {failure_error()}")
elif is_sprt:
    metrics = sprt_metrics()
    if metrics.get("tasks"):
        lines.append(f"  • Tasks: {metrics['tasks']}")
    if metrics.get("games"):
        lines.append(f"  • Games: {metrics['games']}")
    if metrics.get("elo"):
        elo = metrics["elo"]
        if metrics.get("ci"):
            elo = f"{elo} +/- {metrics['ci']}"
        lines.append(f"  • Elo: {elo}")
    if metrics.get("llr"):
        lines.append(f"  • LLR: {metrics['llr']}")
    if metrics.get("los"):
        lines.append(f"  • LOS: {metrics['los']}")
    if metrics.get("draw"):
        lines.append(f"  • Draw: {metrics['draw']}")
    lines.append(f"  • Status: {sprt_status_text()}")
else:
    lines.append(f"  • Status: {final_status() or status or event_name}")

print("\n".join(lines))
PY_RENDER
)

publish() {
    local url="$1"
    local data="$2"
    local title="$3"
    local priority="$4"

    if [ "$DRY_RUN" = "1" ]; then
        printf '%s → %s\n' "$url" "$data"
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

wake_agent() {
    [ "$NOTIFAI_ENABLE" = "1" ] || return 0

    if [ "$DRY_RUN" = "1" ]; then
        printf 'notifai:%s → %s\n' "$NOTIFAI_TARGET" "$rendered"
        return 0
    fi

    if ! command -v "$NOTIFAI_COMMAND" >/dev/null 2>&1; then
        printf '%s event=%s notifai missing command=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$NOTIFAI_COMMAND" >>"$LOG"
        return 1
    fi

    if "$NOTIFAI_COMMAND" "$rendered" "$NOTIFAI_TARGET"; then
        printf '%s event=%s → notifai target=%s\n' \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$NOTIFAI_TARGET" >>"$LOG"
        return 0
    fi

    printf '%s event=%s notifai failed target=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$NOTIFAI_TARGET" >>"$LOG"
    return 1
}

event_selected() {
    local event="$1"
    local list
    list=$(printf '%s' "${2:-}" | tr -d '[:space:]')
    [ -n "$list" ] || return 1

    case ",$list," in
        *,"$event",*) return 0 ;;
    esac
    if [ "$event" = "iteration_done" ]; then
        case ",$list," in
            *,done,*) return 0 ;;
        esac
    fi
    return 1
}

case "$event_name" in
    fail)
        publish "$PING_URL" "$rendered" "Enyo NNUE fail" "5"
        printf '%s event=fail → ping\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
        ;;
    iteration_done)
        publish "$NNUE_URL" "$rendered" "Enyo NNUE iteration done" "4"
        printf '%s event=iteration_done → nnue\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >>"$LOG"
        ;;
    *)
        publish "$AI_STDOUT_URL" "$rendered" "Enyo NNUE $event_name" "3"
        printf '%s event=%s → AI_stdout\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
        ;;
esac

if [ "$AI_STDIN_ENABLE" = "1" ] && event_selected "$event_name" "$AI_STDIN_EVENTS"; then
    if ! wake_agent; then
        priority=4
        [ "$event_name" = "fail" ] && priority=5
        if publish "$AI_STDIN_URL" "$rendered" "Enyo NNUE $event_name" "$priority"; then
            printf '%s event=%s → AI_stdin\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" >>"$LOG"
        else
            rc=$?
            printf '%s event=%s AI_stdin failed rc=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$event_name" "$rc" >>"$LOG"
        fi
    fi
fi
