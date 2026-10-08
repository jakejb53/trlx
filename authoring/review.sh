#!/usr/bin/env bash
# Usage: authoring/review.sh PROMPT_FILE OUT_PREFIX [MODEL] [TIMEOUT_SECONDS]
#
# Runs the adversarial reasoning reviewer as a one-shot CLI process from the
# repository root with the prompt inline, stdin closed, stdout and stderr
# captured separately as OUT_PREFIX.stdout and OUT_PREFIX.stderr. Prints the
# exit status and the final stdout line, which must be exactly ACCEPT or
# REVISE; any other final line is a failed call under DATASET-AUTHORING.md.
set -u
PROMPT="$1"; OUT="$2"; MODEL="${3:-claude-opus-4-8}"; LIMIT="${4:-600}"
cd "$(dirname "$0")/.." || exit 1
timeout "$LIMIT" claude --model "$MODEL" -p "$(cat "$PROMPT")" < /dev/null > "$OUT.stdout" 2> "$OUT.stderr"
code=$?
echo "exit=$code"
echo "ruling=$(tail -n 1 "$OUT.stdout")"
