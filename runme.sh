#!/usr/bin/env bash
set -euo pipefail

build_json=build.json
cmd=all
rows=50000
games=100
concurrency=10
threads=1
pause_between=1

run_override=
prev_override=
engine_override=
candidate_override=
reference_override=
cases_override=

if [[ $# -gt 0 && "$1" != --* ]]; then
  cmd=$1
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) build_json=$2; shift 2 ;;
    --run) run_override=$2; shift 2 ;;
    --prev|--continue-from) prev_override=$2; shift 2 ;;
    --engine) engine_override=$2; shift 2 ;;
    --candidate-net) candidate_override=$2; shift 2 ;;
    --reference-net) reference_override=$2; shift 2 ;;
    --cases) cases_override=$2; shift 2 ;;
    --rows) rows=$2; shift 2 ;;
    --games) games=$2; shift 2 ;;
    --concurrency) concurrency=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --no-pause|-y|--yes) pause_between=0; shift ;;
    -h|--help) cmd=help; shift ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

json_get() {
  jq -r "$1" "$build_json"
}

run=${run_override:-$(json_get '.run')}
prev=${prev_override:-$(json_get '.continue_from // empty')}
engine=${engine_override:-$HOME/assets/engines/enyo_91ede5f}
candidate_net=${candidate_override:-$HOME/assets/nets/$run.nn}
data_file=data/bullet/$run.bullet
run_dir=runs/$run

reference_net=${reference_override:-}
if [[ -z "$reference_net" ]]; then
  if [[ -n "$prev" && "$prev" != "null" ]]; then
    reference_net=$HOME/assets/nets/$prev.nn
  else
    reference_net=$HOME/code/cpp/chess/enyo/net/default.net
  fi
fi

# Manual reference choices. Uncomment exactly one if you want to override the
# default previous-run reference without passing --reference-net.
# reference_net=$HOME/assets/nets/default.net
# reference_net=$HOME/code/cpp/chess/enyo/net/default.net
# reference_net=$HOME/assets/nets/pwa-native-v1.nn

cases=${cases_override:-runs/lichess-inacc-miss-blunder-20260612/native760_vs_native123_net_value_bad_move_gate.jsonl}

mkdir -p "$run_dir"

pause_after() {
  [[ "$pause_between" == 1 ]] || return 0
  [[ -t 0 || -r /dev/tty ]] || return 0
  read -r -p "Press Enter to continue, Ctrl-C to stop: " _ </dev/tty
}

header() {
  printf '\n==== %s ====\n' "$1"
}

show_paths() {
  echo "run=$run"
  echo "prev=$prev"
  echo "engine=$engine"
  echo "reference_net=$reference_net"
  echo "candidate_net=$candidate_net"
  echo "data=$data_file"
}

plan() {
  tools/bullet/train plan --build "$build_json"
}

data() {
  tools/bullet/train data --build "$build_json"
}

train() {
  tools/bullet/train run --build "$build_json"
}

export_net() {
  tools/bullet/train export --build "$build_json"
}

startpos() {
  show_paths
  printf 'uci\nsetoption name nnue_file value %s\nisready\nposition startpos\ngo depth 3\nquit\n' "$candidate_net" \
    | "$engine" \
    | tee "$run_dir/startpos.log"
}

static_eval() {
  show_paths
  .venv/bin/python tools/validate/eval_dataset.py \
    --net "$candidate_net" \
    --data "$data_file" \
    --rows "$rows" \
    --batch-size 4096 \
    --device cpu \
    --buckets \
    | tee "$run_dir/static_eval.txt"

  cat <<'EOF'

MAE   average absolute eval error in centipawns; lower is better
sign  percent of non-zero targets with the correct winner; higher is better
corr  correlation between candidate eval and target; higher is better
bias  average candidate-target error; near 0 is better
slope eval scale relative to target; near 1 is better
EOF
}

move_gate() {
  show_paths
  echo "cases=$cases"
  .venv/bin/python tools/validate/eval_move_gate.py \
    --cases "$cases" \
    --engine "$engine" \
    --reference-net "$reference_net" \
    --candidate-net "$candidate_net" \
    --threads 1 \
    --hash 64 \
    --summary-json "$run_dir/move_gate.summary.json" \
    --output "$run_dir/move_gate.jsonl"

  cat <<'EOF'

Pass shape:
candidate_prefers_best >= reference_prefers_best
fixed > regressed
missing or illegal moves = 0
EOF
}

sprt_100() {
  show_paths
  sprt \
    --concurrency "$concurrency" \
    --threads "$threads" \
    --reference "$engine" \
    --candidate "$engine" \
    --reference-uci "nnue_file=$reference_net" \
    --candidate-uci "nnue_file=$candidate_net" \
    --games "$games" \
    | tee "$run_dir/sprt_${games}.log"
}

phase() {
  header "$1"
  "$1"
  pause_after
}

all() {
  phase plan
  phase data
  phase train
  phase export_net
  phase startpos
  phase static_eval
  phase move_gate
  phase sprt_100
}

help() {
  cat <<EOF
Usage:
  ./runme.sh [phase] [options]

Phases:
  all          run plan, data, train, export, startpos, static_eval, move_gate, sprt_100
  plan         print resolved training plan
  data         convert source binpack to data/bullet/<run>.bullet
  train        run Bullet training
  export       export runs/<run>/model.nn to ~/assets/nets/<run>.nn
  startpos     load candidate net in Enyo and search startpos depth 3
  static_eval  run MAE/sign/corr/bias/slope gate
  move_gate    run selected bad-move gate
  sprt_100     run candidate vs reference SPRT smoke

Options:
  --build FILE          Read this build config instead of build.json.
  --run NAME            Override .run from the build config.
  --prev NAME           Override .continue_from from the build config.
  --continue-from NAME  Same as --prev.
  --engine PATH         Engine binary used by startpos, move_gate, and sprt_100.
  --candidate-net PATH  Candidate .nn file. Defaults to ~/assets/nets/<run>.nn.
  --reference-net PATH  Reference .nn file. Defaults to ~/assets/nets/<prev>.nn.
  --cases PATH          Move-gate JSONL case file.
  --rows N              Rows used by static_eval. Defaults to 50000.
  --games N             Games used by sprt_100. Defaults to 100.
  --concurrency N       SPRT worker concurrency. Defaults to 10.
  --threads N           Engine threads per SPRT game. Defaults to 1.
  --no-pause            Do not stop between phases in all.
  -y, --yes             Same as --no-pause.
  -h, --help            Print this help.

Examples:
  ./runme.sh plan
  ./runme.sh all
  ./runme.sh static_eval --rows 200000
  ./runme.sh sprt_100 --reference-net "\$HOME/assets/nets/default.net" --games 300
EOF
}

case "$cmd" in
  all|plan|data|train|startpos|static_eval|move_gate|sprt_100|help)
    "$cmd"
    ;;
  export)
    export_net
    ;;
  *)
    echo "unknown phase: $cmd" >&2
    help >&2
    exit 2
    ;;
esac
