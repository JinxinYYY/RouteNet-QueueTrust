#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${HERE}/logs"

echo "=== Launcher ==="
if [[ -f "${LOG_ROOT}/latest.pid" ]]; then
    PID="$(tr -d '[:space:]' < "${LOG_ROOT}/latest.pid")"
    if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
        ps -p "${PID}" -o pid=,etime=,stat=,cmd=
    else
        echo "Not running (last PID: ${PID:-unknown})"
    fi
else
    echo "No launcher PID file."
fi

echo
echo "=== Five-seed state ==="
if [[ -f "${HERE}/five_seed_state.json" ]]; then
    cat "${HERE}/five_seed_state.json"
else
    echo "No state file yet."
fi

echo
echo "=== Complete results ==="
find "${HERE}/runs" -type f -name results.json -print 2>/dev/null | sort || true

echo
echo "=== Latest log ==="
if [[ -f "${LOG_ROOT}/latest_log.txt" ]]; then
    LOG_FILE="$(cat "${LOG_ROOT}/latest_log.txt")"
    echo "${LOG_FILE}"
    tail -n 35 "${LOG_FILE}" 2>/dev/null || true
else
    echo "No latest log file."
fi

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

