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

STOCKFISH_NET=${STOCKFISH_NET:-nn-0ee0657fb25e.nnue}
CANDIDATE=${CANDIDATE:-candidate}
GAMES=${GAMES:-500}

REFERENCE_NET=$(resolve_path "$STOCKFISH_NET")
CANDIDATE_NET=$(resolve_path candidate.net)

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
        *)
            echo "Error: Invalid argument '$1'. Only --candidate and --reference flags are allowed." >&2
            exit 2
            ;;
    esac
done

ENGINE="$HOME/assets/engines/$CANDIDATE"
ENGINE_NAME=$(basename "$(readlink -f "$ENGINE")")
BOOK=~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd
RUN="sprt-$(basename "$CANDIDATE_NET")-vs-$(basename "$REFERENCE_NET")-$GAMES-$(date +%Y%m%d-%H%M%S)"
ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
REFERENCE=$(basename "$REFERENCE_NET")
LEDGER=
SPRT_ARGS=()

case "$REFERENCE" in
    *.nnue)      LEDGER=stockfish-net.jsonl ;;
esac

[[ -z $LEDGER ]] || SPRT_ARGS=(--elo0 0 --elo1 10 --alpha 1e-300 --beta 1e-300)

run_sprt() {
    forge sprt \
        --run "$RUN" \
        --wait \
        --verify \
        --comment "candidate=$(basename "$CANDIDATE_NET") vs reference=$(basename "$REFERENCE_NET")" \
        --book "$BOOK" \
        --candidate "$ENGINE" \
        --candidate-uci nnue_file="$CANDIDATE_NET" \
        --concurrency 1 \
        --games "$GAMES" \
        --reference "$ENGINE" \
        --reference-uci nnue_file="$REFERENCE_NET" \
        --restart on \
        --shards 24 \
        --tc 10+0.1 \
        --threads 1 \
        "${SPRT_ARGS[@]}"
}

save_result() {
    [[ -n $LEDGER ]] || return

    mkdir -p "$ROOT/benchmarks"
    forge status "$RUN" --json | jq -c \
        --arg candidate "$(basename "$CANDIDATE_NET" .nn)" \
        --arg engine "$ENGINE_NAME" \
        --arg reference "$REFERENCE" \
        --argjson requested_games "$GAMES" '
        (.progress_fields | map(split("=") | {(.[0]): .[1]}) | add) as $metrics
        | ($metrics.games | split("/") | map(tonumber)) as $games
        | if .completed_at == null or $games[0] != $requested_games then
            error("incomplete Forge result")
          else
        ($metrics.llr | capture("(?<value>-?[0-9.]+)/(?<bound>[0-9.]+)")) as $llr
        | {
            date: .completed_at[0:10],
            candidate: $candidate,
            engine: $engine,
            reference: $reference,
            requested_games: $requested_games,
            games: $games[0],
            elo: ($metrics.elo | tonumber),
            ci: ($metrics.ci | tonumber),
            llr: ($llr.value | tonumber),
            llr_bound: ($llr.bound | tonumber),
            los: ($metrics.los | rtrimstr("%") | tonumber),
            draw: ($metrics.draw | rtrimstr("%") | tonumber)
        }
          end' >>"$ROOT/benchmarks/$LEDGER"
}

main() {
    run_sprt
    save_result

    local result elo llr
    result=$(forge status "$RUN" --json)
    elo=$(jq -r '.display.fields[] | select(startswith("elo=")) | split("=")[1]' <<< "$result")
    llr=$(jq -r '.display.fields[] | select(startswith("llr=")) | split("=")[1] | split("/")[0]' <<< "$result")
    printf 'elo=%s llr=%s\n' "$elo" "$llr"
}

[[ ${BASH_SOURCE[0]} != "$0" ]] || main
