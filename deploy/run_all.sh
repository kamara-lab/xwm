#!/usr/bin/env bash
# Launch every experiment as its own Modal app, concurrently.
#
# One app per experiment means they get separate GPUs and separate volumes, so a
# failure in one does not touch the others, and each can be watched or stopped on
# its own. Logs land in deploy/logs/<experiment>.log.
#
#   ./deploy/run_all.sh              # gpu preset
#   ./deploy/run_all.sh xl           # a bigger preset
set -uo pipefail
PRESET="${1:-gpu}"
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/modal}"
mkdir -p deploy/logs
pids=()
names=()
for app in deploy/app_*.py; do
  name=$(basename "$app" .py | sed 's/^app_//')
  echo "launching $name (preset $PRESET)"
  "$PYTHON" run --detach "$app" --preset "$PRESET" > "deploy/logs/$name.log" 2>&1 &
  pids+=($!)
  names+=("$name")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  ${names[$i]}: ok"
  else
    echo "  ${names[$i]}: FAILED (see deploy/logs/${names[$i]}.log)"
    status=1
  fi
done
exit $status
