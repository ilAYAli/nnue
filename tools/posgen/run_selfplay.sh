#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
assets_dir="$workspace_root/assets"
if [[ ! -d "$assets_dir" && -d "$HOME/code/cpp/chess/assets" ]]; then
    assets_dir="$HOME/code/cpp/chess/assets"
fi

runner=""
engine="$repo_root/build/enyo"
nnue_file=""
games=100
shard_games=0
concurrency=1
threads=1
depth=8
tc=""
book="$assets_dir/books/UHO_Lichess_4852_v1.epd"
book_format="epd"
order="random"
srand=""
restart="off"
output="$repo_root/runs/standalone_posgen/selfplay_$(date +%Y%m%d_%H%M%S).pgn"
extra_engine_options=()

usage() {
    cat <<EOF
Usage: $0 [options]

Generate Enyo-vs-Enyo self-play PGNs with fastchess score comments.

Options:
  --runner PATH               fastchess binary (default: PATH or ~/source/fastchess/fastchess)
  --engine PATH               Engine binary (default: build/enyo)
  --nnue-file PATH            Optional NNUE network path passed as UCI option
  --output PATH               Output PGN (default: runs/standalone_posgen/selfplay_TIMESTAMP.pgn)
  --games N                   Total games to generate (default: 100)
  --shard-games N             Split run into shards of N games (default: disabled)
  --concurrency N             Concurrent games (default: 1)
  --threads N                 UCI Threads per engine (default: 1)
  --depth N                   Fixed search depth when --tc is unset (default: 8)
  --tc TC                     Time control, e.g. 10+0.1; disables fixed depth
  --book PATH                 Opening book/EPD (default: assets UHO book)
  --book-format FORMAT        pgn or epd (default: epd)
  --order ORDER               random or sequential (default: random)
  --srand N                   fastchess random seed
  --restart MODE              fastchess engine restart mode (default: off)
  --engine-option NAME=VALUE  Extra UCI option for both engines; repeatable
  -h, --help                  Show this help
EOF
}

abs_path() {
    python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner) runner="$2"; shift 2 ;;
        --engine) engine="$2"; shift 2 ;;
        --nnue-file) nnue_file="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --games) games="$2"; shift 2 ;;
        --shard-games) shard_games="$2"; shift 2 ;;
        --concurrency) concurrency="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --depth) depth="$2"; shift 2 ;;
        --tc) tc="$2"; shift 2 ;;
        --book) book="$2"; shift 2 ;;
        --book-format) book_format="$2"; shift 2 ;;
        --order) order="$2"; shift 2 ;;
        --srand) srand="$2"; shift 2 ;;
        --restart) restart="$2"; shift 2 ;;
        --engine-option) extra_engine_options+=("option.$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

engine="$(abs_path "$engine")"
output="$(abs_path "$output")"
book="$(abs_path "$book")"
if [[ -n "$runner" ]]; then
    runner="$(abs_path "$runner")"
fi
if [[ -n "$nnue_file" ]]; then
    nnue_file="$(abs_path "$nnue_file")"
fi

if [[ -z "$runner" ]]; then
    if command -v fastchess >/dev/null 2>&1; then
        runner="$(command -v fastchess)"
    elif [[ -x "$HOME/source/fastchess/fastchess" ]]; then
        runner="$HOME/source/fastchess/fastchess"
    elif [[ -x "$assets_dir/../fastchess/fastchess" ]]; then
        runner="$(abs_path "$assets_dir/../fastchess/fastchess")"
    else
        echo "fastchess not found; pass --runner PATH" >&2
        exit 1
    fi
fi

if [[ ! -x "$runner" ]]; then
    echo "fastchess runner is not executable: $runner" >&2
    exit 1
fi
if [[ ! -x "$engine" ]]; then
    echo "Engine is not executable: $engine" >&2
    exit 1
fi
if [[ ! -f "$book" ]]; then
    echo "Opening book not found: $book" >&2
    exit 1
fi
if [[ -n "$nnue_file" && ! -f "$nnue_file" ]]; then
    echo "NNUE file not found: $nnue_file" >&2
    exit 1
fi

mkdir -p "$(dirname "$output")"

case "$restart" in
    auto|on|off) ;;
    *)
        echo "Invalid --restart value: $restart (expected auto, on, or off)" >&2
        exit 2
        ;;
esac

engine_options=("option.Threads=$threads")
if [[ -n "$nnue_file" ]]; then
    engine_options+=("option.nnue_file=$nnue_file")
fi
if ((${#extra_engine_options[@]})); then
    engine_options+=("${extra_engine_options[@]}")
fi

limit_args=()
if [[ -n "$tc" ]]; then
    limit_args=(tc="$tc")
else
    limit_args=(tc=inf depth="$depth")
fi

srand_args=()
if [[ -n "$srand" ]]; then
    srand_args=(-srand "$srand")
fi

echo "selfplay: runner=$runner"
echo "selfplay: engine=$engine"
echo "selfplay: nnue_file=${nnue_file:-<disabled>}"
echo "selfplay: output=$output"
echo "selfplay: games=$games shard_games=$shard_games concurrency=$concurrency threads=$threads restart=$restart limit=${tc:-depth $depth}"

run_fastchess() {
    local shard_output="$1"
    local shard_count="$2"
    local shard_srand="$3"
    local shard_idx="${4:-1}"
    local shard_total="${5:-1}"
    local -a shard_srand_args=()
    if [[ -n "$shard_srand" ]]; then
        shard_srand_args=(-srand "$shard_srand")
    fi

    "$runner" \
        -openings file="$book" format="$book_format" order="$order" \
        -engine name=enyo-a proto=uci restart="$restart" cmd="$engine" "${engine_options[@]}" \
        -engine name=enyo-b proto=uci restart="$restart" cmd="$engine" "${engine_options[@]}" \
        -concurrency "$concurrency" "${shard_srand_args[@]}" \
        -rounds "$shard_count" -games 1 -recover -output format=cutechess \
        -pgnout file="$shard_output" notation=san append=false \
        -each "${limit_args[@]}" \
        -ratinginterval 0 \
        | awk -v total="$shard_count" \
              -v shard="$shard_idx" \
              -v shards="$shard_total" '
            /^Score of / {
                line = $0
                sub(/^Score of [^:]+: /, "", line)
                split(line, parts, " ")
                wins = parts[1]
                losses = parts[3]
                draws = parts[5]
                score = parts[6]
                g = parts[7]
                gsub(/\[/, "", score)
                gsub(/\]/, "", score)
                if (g > 0) {
                    draw_pct = draws * 100.0 / g
                    printf("[%d/%d %4d/%d] selfplay | score %s | draw %5.1f%%\n",
                           shard, shards, g, total, score, draw_pct)
                    fflush()
                }
                next
            }
            /^Elo difference:/ { next }
            /^Started game / { next }
            /^Finished game / { next }
            /^Warning;/ { next }
            /^Info;/ { next }
            /^Position;/ { next }
            /^Moves;/ { next }
            /^Finished match/ { print "selfplay: shard finished"; fflush(); next }
            /^Total Time:/ { print "selfplay: " $0; fflush(); next }
            { print; fflush() }
        '
}

if ((shard_games > 0 && games > shard_games)); then
    output_dir="$(dirname "$output")"
    output_base="$(basename "$output" .pgn)"
    shards=$(((games + shard_games - 1) / shard_games))
    remaining="$games"
    : > "$output"
    for ((shard = 1; shard <= shards; shard++)); do
        count="$shard_games"
        if ((remaining < count)); then
            count="$remaining"
        fi
        shard_output="$output_dir/${output_base}.shard$(printf '%04d' "$shard").pgn"
        shard_srand=""
        if [[ -n "$srand" ]]; then
            shard_srand=$((srand + shard - 1))
        fi
        echo "selfplay: shard $shard/$shards games=$count output=$shard_output"
        run_fastchess "$shard_output" "$count" "$shard_srand" "$shard" "$shards"
        cat "$shard_output" >> "$output"
        remaining=$((remaining - count))
    done
else
    run_fastchess "$output" "$games" "$srand" 1 1
fi

echo "selfplay: wrote $output"
