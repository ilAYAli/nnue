#!/usr/bin/env bash
set -euo pipefail

(( $# >= 1 )) || { echo "Usage: $0 <candidate_net> [reference_net] [games]" >&2; exit 1; }

CANDIDATE=enyo_7483c1c
CANDIDATE_NET="~/assets/nets/$(basename "$1")"
REFERENCE_NET="~/assets/nets/$(basename "${2:-default.net}")"
GAMES=${3:-500}
BOOK=~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd
RUN="sprt-$CANDIDATE-$GAMES-$(date +%Y%m%d-%H%M%S)"
ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
SPRT_ARGS=()

if [[ $REFERENCE_NET == */default.net ]]; then
    SPRT_ARGS=(--elo0 0 --elo1 10 --alpha 1e-300 --beta 1e-300)
fi

run_sprt() {
    forge sprt \
        --run "$RUN" \
        --wait \
        --verify \
        --comment "candidate=$(basename "$CANDIDATE_NET") vs reference=$(basename "$REFERENCE_NET")" \
        --book "$BOOK" \
        --candidate ~/assets/engines/$CANDIDATE \
        --candidate-uci nnue_file="$CANDIDATE_NET" \
        --games "$GAMES" \
        --reference ~/assets/engines/$CANDIDATE \
        --reference-uci nnue_file="$REFERENCE_NET" \
        --restart on \
        --tc 10+0.1 \
        --threads 1 \
        "${SPRT_ARGS[@]}"
}

save_result() {
    [[ $REFERENCE_NET == */default.net ]] || return

    mkdir -p "$ROOT/benchmarks"
    forge status "$RUN" --json | jq -c \
        --arg candidate "$(basename "$CANDIDATE_NET" .nn)" \
        --arg engine "$CANDIDATE" \
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
            reference: "default.net",
            requested_games: $requested_games,
            games: $games[0],
            elo: ($metrics.elo | tonumber),
            ci: ($metrics.ci | tonumber),
            llr: ($llr.value | tonumber),
            llr_bound: ($llr.bound | tonumber),
            los: ($metrics.los | rtrimstr("%") | tonumber),
            draw: ($metrics.draw | rtrimstr("%") | tonumber)
        }
          end' >>"$ROOT/benchmarks/default-net.jsonl"
}

main() {
    run_sprt
    save_result
}

[[ ${BASH_SOURCE[0]} != "$0" ]] || main
