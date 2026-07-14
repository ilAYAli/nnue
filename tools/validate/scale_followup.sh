#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/code/cpp/chess/nnue"
ENGINE="$HOME/code/cpp/chess/enyo/build/enyo"
ORIGINAL="$HOME/assets/nets/enyo-1.28.0-rc16.nn"
BERSERK="$HOME/assets/nets/berserk-9b84c340af7e.nn"
RUN_052="sprt-enyo-scale-root-1.0.0-rc1.nn-vs-enyo-1.28.0-rc16.nn-1500-20260713-234121"
RUN_048="sprt-enyo-scale-root-0.48.0-rc1.nn-vs-enyo-1.28.0-rc16.nn-1500-20260714-003933"
NET_052="$HOME/assets/nets/enyo-scale-root-1.0.0-rc1.nn"
NET_048="$HOME/assets/nets/enyo-scale-root-0.48.0-rc1.nn"
LOG="$ROOT/runs/scale-root-followup-$(date +%Y%m%d-%H%M%S).log"

status_json() {
  forge status "$1" --json 2>/dev/null
}

field_from_json() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
name = sys.argv[2]
fields = payload.get("progress_fields") or payload.get("display", {}).get("fields", [])
for item in fields:
    if item.startswith(name + "="):
        print(item.split("=", 1)[1])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

is_done() {
  python3 - "$1" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print("1" if payload.get("completed_at") else "0")
PY
}

elo_value() {
  local json="$1"
  local text
  text="$(field_from_json "$json" elo)"
  text="${text#+}"
  python3 - "$text" <<'PY'
import sys
print(float(sys.argv[1]))
PY
}

launch_berserk() {
  local net="$1"
  local tag="$2"
  local session="absolute_${tag}_berserk"
  local tmux_tag="${tag//./_}"
  local alt_session="absolute_${tmux_tag}_berserk"
  if tmux has-session -t "$session" 2>/dev/null || tmux has-session -t "$alt_session" 2>/dev/null; then
    echo "$(date -Iseconds) $session already exists" | tee -a "$LOG"
    return
  fi
  mkdir -p "$ROOT/runs/$tag/logs"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && GAMES=1500 tools/validate/sprt_net.sh --candidate '$net' --reference '$BERSERK' --engine '$ENGINE' 2>&1 | tee 'runs/$tag/logs/06-sprt-1500-vs-berserk.log'"
  echo "$(date -Iseconds) launched $session candidate=$net reference=$BERSERK" | tee -a "$LOG"
}

main() {
  cd "$ROOT"
  echo "$(date -Iseconds) follow-up monitor started" | tee -a "$LOG"
  while true; do
    json052="$(status_json "$RUN_052" || true)"
    json048="$(status_json "$RUN_048" || true)"
    if [[ -n "$json052" && -n "$json048" && "$(is_done "$json052")" == "1" && "$(is_done "$json048")" == "1" ]]; then
      elo052="$(elo_value "$json052")"
      elo048="$(elo_value "$json048")"
      echo "$(date -Iseconds) complete 0.52=$elo052 0.48=$elo048" | tee -a "$LOG"
      best_net=""
      best_tag=""
      best_elo="$elo052"
      if python3 - "$elo048" "$elo052" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
      then
        best_elo="$elo048"
        best_net="$NET_048"
        best_tag="enyo-scale-root-0.48.0-rc1"
      else
        best_net="$NET_052"
        best_tag="enyo-scale-root-1.0.0-rc1"
      fi
      if python3 - "$best_elo" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
      then
        launch_berserk "$best_net" "$best_tag"
      else
        echo "$(date -Iseconds) no positive scale-root result; no Berserk launch" | tee -a "$LOG"
      fi
      exit 0
    fi
    echo "$(date -Iseconds) waiting for scale-root SPRTs" | tee -a "$LOG"
    sleep 60
  done
}

main "$@"
