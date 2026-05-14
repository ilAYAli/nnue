#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/nnue2/train_new_net_pwa.sh [options]

Launch a complete Enyo NNUE training run on pwa-5090.

Default recipe:
  8M Stockfish-labeled self-play rows
  2M Lichess eval-DB rows
  MPE25 objective, cautious LR, no binpack
  static eval, then SPRT versus the current reference net

Options:
  --session NAME        tmux session to use (default: nnue_test)
  --foreground          run in the current shell instead of tmux
  --allow-active        do not refuse when another NNUE job is active
  --dry-run             print the command without running it
  --selfplay-rows N     default: 8000000
  --lichess-rows N      default: 2000000
  --epochs N            default: 4
  --lr VALUE            default: 3e-7
  --target-clamp N      default: 1200
  --sprt-games N        default: 4000
  --sprt-tc TC          default: 2+0.02
  --sprt-concurrency N  default: 10
  --sprt-threads N      default: 2
  --sprt-hash MB        default: 512
  --sprt-elo1 N         default: 8
  -h, --help            show this help

Examples:
  tools/nnue2/train_new_net_pwa.sh
  tools/nnue2/train_new_net_pwa.sh --selfplay-rows 12000000 --lichess-rows 3000000
  tools/nnue2/train_new_net_pwa.sh --foreground
EOF
}

repo="${NNUE_REPO:-$HOME/code/cpp/chess/nnue}"
session="${SESSION:-nnue_test}"
foreground=0
allow_active=0
dry_run=0

selfplay_rows="${SELFPLAY_ROWS:-8000000}"
lichess_rows="${LICHESS_ROWS:-2000000}"
epochs="${TRAIN_EPOCHS:-4}"
lr="${TRAIN_LR:-3e-7}"
target_clamp="${TARGET_CLAMP:-1200}"
sprt_games="${SPRT_GAMES:-4000}"
sprt_tc="${SPRT_TC:-2+0.02}"
sprt_concurrency="${SPRT_CONCURRENCY:-10}"
sprt_threads="${SPRT_THREADS:-2}"
sprt_hash="${SPRT_HASH:-512}"
sprt_elo1="${SPRT_ELO1:-8}"

while [ "$#" -gt 0 ]; do
  case "$1" in
  --session)
    session="$2"
    shift 2
    ;;
  --foreground)
    foreground=1
    shift
    ;;
  --allow-active)
    allow_active=1
    shift
    ;;
  --dry-run)
    dry_run=1
    shift
    ;;
  --selfplay-rows)
    selfplay_rows="$2"
    shift 2
    ;;
  --lichess-rows)
    lichess_rows="$2"
    shift 2
    ;;
  --epochs)
    epochs="$2"
    shift 2
    ;;
  --lr)
    lr="$2"
    shift 2
    ;;
  --target-clamp)
    target_clamp="$2"
    shift 2
    ;;
  --sprt-games)
    sprt_games="$2"
    shift 2
    ;;
  --sprt-tc)
    sprt_tc="$2"
    shift 2
    ;;
  --sprt-concurrency)
    sprt_concurrency="$2"
    shift 2
    ;;
  --sprt-threads)
    sprt_threads="$2"
    shift 2
    ;;
  --sprt-hash)
    sprt_hash="$2"
    shift 2
    ;;
  --sprt-elo1)
    sprt_elo1="$2"
    shift 2
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if [ ! -d "$repo" ]; then
  echo "missing NNUE repo: $repo" >&2
  exit 1
fi

active_pattern='run_selflichess_mix_pwa\.sh|tools/nnue2/train\.py|tools/nnue2/train_pairwise\.py|/sprt |fastchess'
if [ "$allow_active" -eq 0 ] && [ "$dry_run" -eq 0 ]; then
  active="$(pgrep -af "$active_pattern" | grep -v 'train_new_net_pwa.sh' || true)"
  if [ -n "$active" ]; then
    echo "refusing to start because an NNUE job appears active:" >&2
    echo "$active" >&2
    echo "Use --allow-active only if you intentionally want overlap." >&2
    exit 1
  fi
fi

run="${RUN:-$HOME/tmp/enyo_teacher/selflichess_mix_$(date +%Y%m%d_%H%M%S)}"
cmd=(
  env
  "CODEX_TMUX=$session"
  "RUN=$run"
  "SELFPLAY_ROWS=$selfplay_rows"
  "LICHESS_ROWS=$lichess_rows"
  "TRAIN_EPOCHS=$epochs"
  "TRAIN_LR=$lr"
  "TARGET_CLAMP=$target_clamp"
  "SPRT_GAMES=$sprt_games"
  "SPRT_TC=$sprt_tc"
  "SPRT_CONCURRENCY=$sprt_concurrency"
  "SPRT_THREADS=$sprt_threads"
  "SPRT_HASH=$sprt_hash"
  "SPRT_ELO1=$sprt_elo1"
  tools/nnue2/run_selflichess_mix_pwa.sh
)

printf -v cmdline "%q " "${cmd[@]}"
full_cmd="cd $(printf '%q' "$repo") && $cmdline"

echo "run: $run"
echo "tmux session: $session"
echo "command: $full_cmd"

if [ "$dry_run" -eq 1 ]; then
  exit 0
fi

if [ "$foreground" -eq 1 ]; then
  cd "$repo"
  exec "${cmd[@]}"
fi

tmux has-session -t "$session" 2>/dev/null || \
  tmux new-session -d -s "$session" -n main
tmux send-keys -t "$session" "$full_cmd" C-m
echo "started in tmux session '$session'"
echo "attach with: tmux attach -t $session"
