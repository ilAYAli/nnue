#!/usr/bin/env bash
set -euo pipefail
# ~/assets/nets/native-3.15.0-rc1.nn
#--reference-uci "nnue_file=~/assets/nets/default.net" \

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <candidate_net_path> [reference_net_path] [games]" >&2
    exit 1
fi

exec 9>/tmp/sprt_default_net.lock
if ! flock -n 9; then
    echo "Error: another default-net SPRT is already running" >&2
    exit 1
fi

STATUS_JSON=$(crucible status --json)
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
    echo "Error: Crucible is busy: $BUSY_RUN" >&2
    exit 1
fi

CANDIDATE_NET="$1"
REFERENCE_NET="${2:-~/assets/nets/native-4.0.0-rc2.nn}"
GAMES="${3:-200}"

if [[ "$CANDIDATE_NET" = /* ]]; then
    echo "Warning: candidate net path is absolute: $CANDIDATE_NET" >&2
fi

# Expand ~ manually for validation
CANDIDATE_NET_EXPANDED="${CANDIDATE_NET/#\~/$HOME}"
REFERENCE_NET_EXPANDED="${REFERENCE_NET/#\~/$HOME}"

if [[ ! -f "$CANDIDATE_NET_EXPANDED" ]]; then
    echo "Error: candidate net not found: $CANDIDATE_NET" >&2
    exit 1
fi

if [[ ! -f "$REFERENCE_NET_EXPANDED" ]]; then
    echo "Error: reference net not found: $REFERENCE_NET" >&2
    exit 1
fi

OUTPUT=$(mktemp)
trap 'rm -f "$OUTPUT"' EXIT

set -x
crucible \
    sprt \
    --comment "candidate=$(basename $CANDIDATE_NET) vs reference=$(basename $REFERENCE_NET)" \
    --concurrency 1 \
    --threads 1 \
    --book ~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd \
    --reference "~/assets/engines/enyo_085acb7" \
    --candidate "~/assets/engines/enyo_085acb7" \
    --reference-uci "nnue_file=$REFERENCE_NET" \
    --candidate-uci "nnue_file=$CANDIDATE_NET" \
    --games "$GAMES" 2>&1 | tee "$OUTPUT"

set +x

RUN=$(sed -n 's/^run: id=//p' "$OUTPUT" | tail -1)
if [[ -z "$RUN" ]]; then
    echo "Error: could not determine Crucible run ID" >&2
    exit 1
fi

MANIFEST="$HOME/code/cpp/chess/crucible/runs/$RUN/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: missing Crucible manifest: $MANIFEST" >&2
    exit 1
fi

crucible wait --manifest "$MANIFEST" --interval-seconds 5 --timeout-seconds 0
crucible status "$RUN" --verbose
