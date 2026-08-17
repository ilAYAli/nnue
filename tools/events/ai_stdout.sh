#!/usr/bin/env bash
set -euo pipefail

# Publish concise agent-visible output. This is for summaries, conclusions,
# actions taken, and next steps. Raw commands and long logs should stay in the
# tmux pane or run log.

URL=${NNUE_AI_STDOUT_URL:-https://ntfy.wahlman.no/llmsh}
TITLE=${NNUE_AI_STDOUT_TITLE:-Enyo NNUE agent}
PRIORITY=${NNUE_AI_STDOUT_PRIORITY:-4}
DRY_RUN=${NNUE_AI_STDOUT_DRY_RUN:-0}

if [ "$#" -gt 0 ]; then
    body=$*
else
    body=$(cat)
fi

if [ -z "${body//[[:space:]]/}" ]; then
    echo "ai_stdout.sh: empty message" >&2
    exit 2
fi

if [ "$DRY_RUN" = "1" ]; then
    printf '%s\n' "$body"
    exit 0
fi

source "$HOME/.ntfy" 2>/dev/null || true

if [ -n "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" ]; then
    curl -fsS -m 10 -u "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" \
        -H "Title: $TITLE" -H "Priority: $PRIORITY" -H "Tags: ai-out" \
        --data-binary "$body" "$URL" >/dev/null
else
    curl -fsS -m 10 \
        -H "Title: $TITLE" -H "Priority: $PRIORITY" -H "Tags: ai-out" \
        --data-binary "$body" "$URL" >/dev/null
fi
