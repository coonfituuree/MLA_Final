#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --index-strategy unsafe-best-match --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
npm --prefix frontend ci
echo 'Setup complete. Next: .venv/bin/python scripts/download_data.py && .venv/bin/python -m training.train'
