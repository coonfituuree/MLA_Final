#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -x .venv/bin/python ]] || { echo 'Run ./scripts/setup.sh first.' >&2; exit 1; }
[[ -f artifacts/emscad-v1/manifest.json ]] || { echo 'Original v1 bundle is required; restore it before reproducing v2.' >&2; exit 1; }
.venv/bin/python scripts/download_data.py
.venv/bin/python -m training.experiments
TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m training.train_encoder
.venv/bin/python -m training.package_v2
.venv/bin/python -m diagnostics.audit
.venv/bin/python -m diagnostics.audit_real
.venv/bin/python -m diagnostics.compare_errors
.venv/bin/python scripts/summarize_improvements.py
.venv/bin/python scripts/update_slides_v2.py
.venv/bin/python scripts/browser_fixture.py
echo 'v2 trained and evaluated. Restart the local application with ./scripts/start.sh'
