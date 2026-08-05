#!/usr/bin/env bash
set -euo pipefail

resolve_path() {
    local path="$1"
    if [[ "$path" =~ ^(/|\./|\.\./) ]]; then
        echo "$path"
    else
        echo "$HOME/assets/nets/$path"
    fi
}

ENGINE=${ENGINE:-reference}
REFERENCE_NET=${REFERENCE_NET:-nn-0ee0657fb25e.nnue}
CANDIDATE_NET=${CANDIDATE_NET:-candidate.net}
GAMES=${GAMES:-1500}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference)
            REFERENCE_NET=$(resolve_path "$2")
            shift 2
            ;;
        --candidate)
            CANDIDATE_NET=$(resolve_path "$2")
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        *)
            echo "Error: Invalid argument '$1'. Only --candidate, --reference, and --engine flags are allowed." >&2
            exit 2
            ;;
    esac
done

REFERENCE_NET=$(resolve_path "$REFERENCE_NET")
CANDIDATE_NET=$(resolve_path "$CANDIDATE_NET")

if [[ "$ENGINE" =~ ^(/|\./|\.\./) ]]; then
    :
else
    ENGINE="$HOME/assets/engines/$ENGINE"
fi
ENGINE_NAME=$(basename "$(readlink -f "$ENGINE")")
BOOK=~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd
ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
REFERENCE_NAME=$(basename "$REFERENCE_NET")
DB="$ROOT/benchmarks/benchmarks.db"
SPRT_ARGS=(--elo0 0 --elo1 10 --alpha 1e-300 --beta 1e-300)
RUN=

check_engine_loads_net() {
    local role="$1"
    local net="$2"
    local output rc=0

    if command -v timeout >/dev/null 2>&1; then
        output=$(printf 'setoption name nnue_file value %s\nquit\n' "$net" | timeout 20 "$ENGINE" 2>&1) || rc=$?
    elif command -v gtimeout >/dev/null 2>&1; then
        output=$(printf 'setoption name nnue_file value %s\nquit\n' "$net" | gtimeout 20 "$ENGINE" 2>&1) || rc=$?
    else
        output=$(printf 'setoption name nnue_file value %s\nquit\n' "$net" | "$ENGINE" 2>&1) || rc=$?
    fi

    local resolved_net
    resolved_net=$(readlink -f "$net" 2>/dev/null || printf '%s' "$net")
    if (( rc != 0 )) || ! grep -Fq "path='$resolved_net'" <<<"$output" \
        || grep -Eq 'ERROR:|falling back' <<<"$output"; then
        echo "Error: ENGINE cannot load $role: engine=$ENGINE net=$net" >&2
        printf '%s\n' "$output" | tail -40 >&2
        exit 1
    fi
}

# Deploys via the async `forge run sprt` template (not the blocking `forge sprt`
# helper) with HOOK_EVENTS set, so the globally-configured notify_command
# (~/code/chess/forge/scripts/forge_event_ntfy.sh) fires real done/fail
# notifications for this run - matching how every other Forge job in this
# project reports progress, instead of silently blocking with no visibility.
run_sprt() {
    local deploy_output
    deploy_output=$(HOOK_EVENTS=done,fail forge run sprt \
        --comment "candidate=$(basename "$CANDIDATE_NET") vs reference=$(basename "$REFERENCE_NET")" \
        --reference "$ENGINE" \
        --candidate "$ENGINE" \
        --reference-net "$REFERENCE_NET" \
        --candidate-net "$CANDIDATE_NET" \
        --restart on \
        --games "$GAMES" \
        "${SPRT_ARGS[@]}")
    printf '%s\n' "$deploy_output"

    RUN=$(grep -m1 '^run: id=' <<<"$deploy_output" | sed 's/^run: id=//')
    [[ -n $RUN ]] || { echo "Error: could not parse run id from forge run sprt output" >&2; exit 1; }

    # Deploy itself is async (returns as soon as workers are launched); block
    # here so save_result() below only runs once the SPRT has actually finished.
    HOOK_EVENTS=done,fail forge resume "$RUN" --wait --verify --timeout-seconds 0
}

save_result() {
    mkdir -p "$ROOT/benchmarks"
    local status_file candidate
    status_file=$(mktemp)
    trap 'rm -f "$status_file"' RETURN
    forge status "$RUN" --json >"$status_file"

    candidate=$(basename "$(readlink -f "$CANDIDATE_NET")" .nn)

    python3 - "$DB" "$ENGINE_NAME" "$REFERENCE_NAME" "$GAMES" "$candidate" "$status_file" <<'PY'
import json
import sqlite3
import sys

db_path, engine_name, reference_name, requested_games, candidate, status_file = sys.argv[1:7]
with open(status_file) as f:
    status = json.load(f)

if not status.get("completed_at"):
    sys.exit("Error: incomplete Forge result")

metrics = {}
for field in status.get("display", {}).get("fields", []):
    key, _, value = field.partition("=")
    metrics[key] = value

games = int(metrics["games"].split("/")[0])
requested_games = int(requested_games)
if games != requested_games:
    sys.exit(f"Error: incomplete Forge result (games={games} requested={requested_games})")

llr_value, _, llr_rest = metrics["llr"].partition("/")
llr_bound = llr_rest.split(" ")[0] if llr_rest else None

def pct(key):
    value = metrics.get(key)
    if not value:
        return None
    return float(value.rstrip("%"))

conn = sqlite3.connect(db_path)
conn.execute(
    """
    INSERT INTO benchmark
        (date, candidate, engine, reference_net, requested_games, games,
         elo, ci, llr, llr_bound, los, draw, source_ledger, raw_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        status["completed_at"][:10],
        candidate,
        engine_name,
        reference_name,
        requested_games,
        games,
        float(metrics["elo"]),
        float(metrics["ci"]) if metrics.get("ci") else None,
        float(llr_value) if llr_value else None,
        float(llr_bound) if llr_bound else None,
        pct("los"),
        pct("draw"),
        "sprt_net.sh",
        json.dumps(status),
    ),
)
conn.commit()
print(f"recorded: candidate={candidate} reference_net={reference_name} elo={metrics['elo']}")
PY
}

main() {
    [[ -x $ENGINE ]] || { echo "Error: ENGINE is not executable: $ENGINE" >&2; exit 1; }
    [[ -f $CANDIDATE_NET ]] || { echo "Error: CANDIDATE_NET not found: $CANDIDATE_NET" >&2; exit 1; }
    [[ -f $REFERENCE_NET ]] || { echo "Error: REFERENCE_NET not found: $REFERENCE_NET" >&2; exit 1; }
    check_engine_loads_net CANDIDATE_NET "$CANDIDATE_NET"
    check_engine_loads_net REFERENCE_NET "$REFERENCE_NET"
    run_sprt
    save_result

    local result elo llr
    result=$(forge status "$RUN" --json)
    elo=$(jq -r '.display.fields[] | select(startswith("elo=")) | split("=")[1]' <<< "$result")
    llr=$(jq -r '.display.fields[] | select(startswith("llr=")) | split("=")[1] | split("/")[0]' <<< "$result")
    printf 'elo=%s llr=%s\n' "$elo" "$llr"
}

[[ ${BASH_SOURCE[0]} != "$0" ]] || main
