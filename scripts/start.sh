#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python && -d frontend/node_modules ]] || { echo 'Run ./scripts/setup.sh first.' >&2; exit 1; }
[[ -f artifacts/current.txt ]] || { echo 'No trained bundle. Run .venv/bin/python -m training.train first.' >&2; exit 1; }
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
pids+=("$!")
for attempt in {1..45}; do
  if curl --silent --fail http://127.0.0.1:8000/api/ready >/dev/null; then break; fi
  kill -0 "${pids[0]}" 2>/dev/null || { echo 'Backend exited. Check the port and model bundle.' >&2; exit 1; }
  if [[ "$attempt" == 45 ]]; then echo 'Backend did not become ready. See model loading error above.' >&2; exit 1; fi
  sleep 1
done
node frontend/node_modules/vite/bin/vite.js frontend --host localhost --port 5173 --strictPort &
pids+=("$!")
echo 'Frontend: http://localhost:5173 | Backend: http://127.0.0.1:8000'
wait -n "${pids[@]}"
