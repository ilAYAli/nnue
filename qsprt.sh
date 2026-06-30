#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# ~/assets/nets/native-3.15.0-rc1.nn
#--reference-uci "nnue_file=~/assets/nets/default.net" \

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <candidate_net_path> [reference_net_path] [games]" >&2
    exit 1
fi

CANDIDATE_NET="$1"
REFERENCE_NET="${2:-~/assets/nets/native-4.0.0-rc2.nn}"
GAMES="${3:-1500}"
CANDIDATE_FILE=$(basename "$CANDIDATE_NET")
CANDIDATE=${CANDIDATE_FILE%.nn}
OUTPUT=
EVENT_SENT=0

notify_ai() {
    local event="$1"
    local status="$2"
    local message="$3"
    EVENT_SENT=1
    jq -cn \
        --arg event "$event" \
        --arg run "$CANDIDATE" \
        --arg stage qsprt \
        --arg status "$status" \
        --arg message "$message" \
        '{event:$event,run:$run,stage:$stage,status:$status,message:$message}' \
        | NNUE_AI_STDIN_EVENTS=done,fail \
            "$SCRIPT_DIR/tools/events/nnue_event_ntfy.sh" || true
}

on_exit() {
    local rc=$?
    trap - EXIT
    [[ -z "$OUTPUT" ]] || rm -f "$OUTPUT"
    if ((EVENT_SENT == 0)); then
        if ((rc == 0)); then
            notify_ai done done \
                "qsprt completed: $CANDIDATE vs $(basename "$REFERENCE_NET"), games=$GAMES"
        else
            notify_ai fail fail \
                "qsprt failed: $CANDIDATE vs $(basename "$REFERENCE_NET"), rc=$rc"
        fi
    fi
    exit "$rc"
}
trap on_exit EXIT

exec 9>/tmp/sprt_default_net.lock
if ! flock -n 9; then
    echo "Error: another default-net SPRT is already running" >&2
    exit 1
fi

STATUS_JSON=$(forge status --json)
BUSY_RUN=$(jq -r '
    def inactive:
        (.state // "") as $state
        | $state == "done" or $state == "ok" or $state == "completed"
          or $state == "stopped" or $state == "failed" or $state == "fail";
    (.runs // [])
    | map(select((inactive | not) and ((.tasks // 0) > 0)
                 and ((.done // 0) < (.tasks // 0))))
    | .[0].run // empty
' <<< "$STATUS_JSON")
if [[ -n "$BUSY_RUN" ]]; then
    echo "Error: forge is busy: $BUSY_RUN" >&2
    exit 1
fi

# Expand ~ manually for validation
CANDIDATE_NET_EXPANDED="${CANDIDATE_NET/#\~/$HOME}"
REFERENCE_NET_EXPANDED="${REFERENCE_NET/#\~/$HOME}"

SPRT_EXTRA_ARGS=()
if [[ "$(basename "$REFERENCE_NET_EXPANDED")" == "default.net" ]]; then
    # Keep SPRT reporting, but make its bounds unreachable for a fixed-game
    # checkpoint. Forge and every shard therefore run the full game count.
    SPRT_EXTRA_ARGS=(--elo0 0 --elo1 10 --alpha 1e-300 --beta 1e-300)
fi

if [[ ! -f "$CANDIDATE_NET_EXPANDED" ]]; then
    echo "Error: candidate net not found: $CANDIDATE_NET" >&2
    exit 1
fi

if [[ ! -f "$REFERENCE_NET_EXPANDED" ]]; then
    echo "Error: reference net not found: $REFERENCE_NET" >&2
    exit 1
fi

portable_home_path() {
    local path="$1"

    if [[ "$path" != "$HOME/"* ]]; then
        echo "Error: distributed net path is outside HOME: $path" >&2
        return 1
    fi

    printf '~/%s\n' "${path#"$HOME/"}"
}

# Workers may have different home directories, so UCI paths must remain
# home-relative after validation on the coordinator.
CANDIDATE_NET_UCI=$(portable_home_path "$CANDIDATE_NET_EXPANDED")
REFERENCE_NET_UCI=$(portable_home_path "$REFERENCE_NET_EXPANDED")

OUTPUT=$(mktemp)

set -x
forge \
    sprt \
    --comment "candidate=$(basename $CANDIDATE_NET) vs reference=$(basename $REFERENCE_NET)" \
    --concurrency 1 \
    --threads 1 \
    --book ~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd \
    --reference "~/assets/engines/enyo_91fd903" \
    --candidate "~/assets/engines/enyo_91fd903" \
    --reference-uci "nnue_file=$REFERENCE_NET_UCI" \
    --candidate-uci "nnue_file=$CANDIDATE_NET_UCI" \
    "${SPRT_EXTRA_ARGS[@]}" \
    --games "$GAMES" 2>&1 | tee "$OUTPUT"

set +x

RUN=$(sed -n 's/^run: id=//p' "$OUTPUT" | tail -1)
if [[ -z "$RUN" ]]; then
    echo "Error: could not determine forge run ID" >&2
    exit 1
fi

MANIFEST="$HOME/code/cpp/chess/forge/runs/$RUN/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: missing forge manifest: $MANIFEST" >&2
    exit 1
fi

forge wait --manifest "$MANIFEST" --interval-seconds 5 --timeout-seconds 0

STATUS_JSON=$(forge status "$RUN" --json)
if [[ "$(basename "$REFERENCE_NET_EXPANDED")" == "default.net" ]]; then
    if ! jq -e '.state == "done" or .state == "ok" or .state == "completed"' \
        >/dev/null <<< "$STATUS_JSON"
    then
        echo "Error: refusing to log incomplete Forge run: $RUN" >&2
        exit 1
    fi

    ACTUAL_GAMES=$(jq -r '
        [.progress_fields[] | select(startswith("games="))][0]
        | sub("^games="; "") | split("/")[0]
    ' <<< "$STATUS_JSON")
    if [[ "$ACTUAL_GAMES" != "$GAMES" ]]; then
        echo "Error: refusing to log partial default-net result: $ACTUAL_GAMES/$GAMES games" >&2
        exit 1
    fi

    LEDGER="$SCRIPT_DIR/benchmarks/default-net.jsonl"
    mkdir -p "$(dirname "$LEDGER")"
    jq -c \
        --arg candidate "$CANDIDATE" \
        --arg engine "enyo_91fd903" \
        --arg reference "default.net" \
        --arg requested_games "$GAMES" '
            def value($name):
                ([.progress_fields[] | select(startswith($name + "="))][0]
                 | sub("^[^=]+="; ""));
            (value("games") | split("/")) as $games
            | (value("llr") | split(" ")[0] | split("/")) as $llr
            | {
                date: (.completed_at | split("T")[0]),
                candidate: $candidate,
                engine: $engine,
                reference: $reference,
                requested_games: ($requested_games | tonumber),
                games: ($games[0] | tonumber),
                elo: (value("elo") | tonumber),
                ci: (value("ci") | tonumber),
                llr: ($llr[0] | tonumber),
                llr_bound: ($llr[1] | tonumber),
                los: (value("los") | sub("%$"; "") | tonumber),
                draw: (value("draw") | sub("%$"; "") | tonumber)
            }
        ' <<< "$STATUS_JSON" >> "$LEDGER"
    echo "Logged default-net result: $LEDGER"
fi

forge status "$RUN" --verbose
