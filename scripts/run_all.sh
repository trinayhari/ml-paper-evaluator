#!/usr/bin/env bash
set -euo pipefail

VENUES=(ICLR.cc/2023/Conference ICLR.cc/2024/Conference)
mkdir -p data/raw data/processed data/pdfs data/splits results
PYTHONPATH=. python scripts/01_collect_openreview.py --venues "${VENUES[@]}" --out data/raw/openreview_notes.jsonl
PYTHONPATH=. python scripts/02_extract_pdfs.py --input data/raw/openreview_notes.jsonl --out data/processed/papers.jsonl --pdf-dir data/pdfs
PYTHONPATH=. python scripts/03_make_splits.py --input data/processed/papers.jsonl --out-dir data/splits --test-venues ICLR.cc/2024/Conference
PYTHONPATH=. python scripts/04_train_baselines.py --train data/splits/train.jsonl --dev data/splits/dev.jsonl --test data/splits/test.jsonl --out results/baselines
PYTHONPATH=. python scripts/05_build_retrieval_index.py --train data/splits/train.jsonl --out results/retrieval
