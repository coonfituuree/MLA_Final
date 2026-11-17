#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
.venv/bin/python -m pytest -q
.venv/bin/python -m diagnostics.audit
.venv/bin/python -m diagnostics.audit_real
.venv/bin/python -m diagnostics.compare_errors
.venv/bin/python scripts/summarize_improvements.py
.venv/bin/python scripts/update_slides_v2.py
.venv/bin/python scripts/browser_fixture.py
npm --prefix frontend run typecheck
npm --prefix frontend run build
echo 'Python, diagnostics, artifact and build checks passed. Start the app and run npm --prefix frontend run test:e2e for browser verification.'
