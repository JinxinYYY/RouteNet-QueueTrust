#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$(cd "${HERE}/../../../RouteNet-Fermi" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${SOURCE_ROOT}/.conda-env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

LOG_ROOT="${HERE}/logs"
mkdir -p "${LOG_ROOT}"
LATEST_PID_FILE="${LOG_ROOT}/latest.pid"

if [[ -f "${LATEST_PID_FILE}" ]]; then
    OLD_PID="$(tr -d '[:space:]' < "${LATEST_PID_FILE}")"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "The latest-model five-seed launcher is already running: PID ${OLD_PID}" >&2
        echo "Use: bash '${HERE}/status_five_seeds.sh'" >&2
        exit 2
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/five_seeds_${TIMESTAMP}.log"
PID_FILE="${LOG_ROOT}/five_seeds_${TIMESTAMP}.pid"

nohup "${PYTHON_BIN}" -u "${HERE}/run_five_seeds.py" \
    --source-root "${SOURCE_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    "$@" \
    > "${LOG_FILE}" 2>&1 < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${PID_FILE}"
printf '%s\n' "${PID}" > "${LATEST_PID_FILE}"
printf '%s\n' "${LOG_FILE}" > "${LOG_ROOT}/latest_log.txt"

echo "Latest reliability-gated model: five-seed training started."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Status: bash '${HERE}/status_five_seeds.sh'"
echo "Follow: tail -f '${LOG_FILE}'"

