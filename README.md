# OpenReview Acceptance Predictor

End-to-end NLP final project scaffold for predicting ML paper accept/reject outcomes from paper text alone, using OpenReview data.

## What this project includes

- OpenReview data collection for ICLR-style venues
- PDF downloading and text extraction
- Clean train/dev/test split by venue/year
- Baselines:
  - TF-IDF + Logistic Regression
  - SentenceTransformer embeddings + Logistic Regression
- Proposed model:
  - Retrieval-augmented LLM prompting over similar historical papers
  - Optional LoRA fine-tuning script for an open-source causal LM
- Evaluation:
  - accuracy, precision, recall, F1, AUROC
  - calibration curve and expected calibration error
- Structured feedback generation prompt
- 4-page report starter in `reports/final_report.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional OpenReview login is only needed for non-public/private venues. Public ICLR data is usually accessible without credentials.

```bash
export OPENREVIEW_USERNAME="your_username"
export OPENREVIEW_PASSWORD="your_password"
```

## Default Usage

If you already have `data/processed/papers.jsonl`, the default path is the Colab notebook or the single pipeline wrapper. This is the recommended way to run the project for training, retrieval, RAG, and optional LoRA without manually coordinating intermediate scripts.

Notebook:

`notebooks/openreview_colab_pipeline.ipynb`

One-command wrapper:

```bash
bash scripts/run_colab_pipeline.sh \
  --input /content/drive/MyDrive/openreview/papers.jsonl \
  --output-root /content/drive/MyDrive/openreview/colab_pipeline_run \
  --clean-output
```

What this handles automatically:

- train/dev/test split creation
- baseline training
- retrieval index build
- RAG prediction
- optional LoRA fine-tuning

## Data Creation

Use the following steps only if you need to rebuild `data/processed/papers.jsonl` from OpenReview and paper PDFs.

```bash
# 1. Collect metadata and decisions
python scripts/01_collect_openreview.py --venues ICLR.cc/2023/Conference ICLR.cc/2024/Conference --out data/raw/openreview_notes.jsonl

# 2. Download PDFs and extract text
python scripts/02_extract_pdfs.py --input data/raw/openreview_notes.jsonl --out data/processed/papers.jsonl --pdf-dir data/pdfs

```

## Colab One-Run Notebook Flow

If you already have `data/processed/papers.jsonl`, use the Colab notebook at `notebooks/openreview_colab_pipeline.ipynb`.

It is built around a single pipeline entrypoint:

```bash
bash scripts/run_colab_pipeline.sh --input /content/drive/MyDrive/openreview/papers.jsonl
```

What it does automatically:

- creates train/dev/test splits
- trains baselines
- builds the retrieval index
- runs RAG prediction on a Colab-sized default sample
- optionally runs LoRA fine-tuning if you pass `--run-lora`

Useful options:

```bash
# clean old outputs and run the default Colab pipeline
bash scripts/run_colab_pipeline.sh \
  --input /content/drive/MyDrive/openreview/papers.jsonl \
  --clean-output

# override the output directory and RAG sample size
bash scripts/run_colab_pipeline.sh \
  --input /content/drive/MyDrive/openreview/papers.jsonl \
  --output-root /content/drive/MyDrive/openreview_run \
  --rag-limit 200

# include LoRA fine-tuning
bash scripts/run_colab_pipeline.sh \
  --input /content/drive/MyDrive/openreview/papers.jsonl \
  --run-lora
```

## Project claim to defend

The cleanest claim is not “we can replace reviewers.” It is:

> Historical paper text contains enough signal to build a moderately accurate pre-review acceptance predictor, and retrieval-augmented context improves interpretability by grounding predictions in similar prior accepted/rejected papers.

## Important caveat

Do not train on review text if the goal is pre-submission prediction. Reviews, ratings, rebuttals, and discussion threads are leakage because authors do not have them before submission.
