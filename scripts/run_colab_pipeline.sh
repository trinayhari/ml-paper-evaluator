#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/08_run_colab_pipeline.py "$@"
