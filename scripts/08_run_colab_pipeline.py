import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl


def run_step(args):
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not pythonpath else f"{ROOT}:{pythonpath}"
    print(f"\n==> {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def ensure_exists(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def count_rows(path: Path) -> int:
    return len(read_jsonl(str(path)))


def main():
    parser = argparse.ArgumentParser(
        description="Single-entry pipeline for Colab runs starting from processed papers.jsonl."
    )
    parser.add_argument("--input", required=True, help="Path to processed papers.jsonl")
    parser.add_argument(
        "--output-root",
        default="artifacts/colab_pipeline",
        help="Directory where splits, results, and optional models will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-venues", nargs="*", default=[])
    parser.add_argument("--rag-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--rag-limit", type=int, default=100)
    parser.add_argument("--rag-k", type=int, default=5)
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--run-lora", action="store_true")
    parser.add_argument("--lora-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    ensure_exists(input_path, "Processed dataset")

    output_root = Path(args.output_root).resolve()
    splits_dir = output_root / "splits"
    baselines_dir = output_root / "baselines"
    retrieval_dir = output_root / "retrieval"
    rag_out = output_root / "rag_predictions.jsonl"
    lora_dir = output_root / "lora_model"
    summary_path = output_root / "run_summary.json"

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_cmd = [
        sys.executable,
        "scripts/03_make_splits.py",
        "--input",
        str(input_path),
        "--out-dir",
        str(splits_dir),
        "--seed",
        str(args.seed),
    ]
    if args.test_venues:
        split_cmd.extend(["--test-venues", *args.test_venues])
    run_step(split_cmd)

    train_path = splits_dir / "train.jsonl"
    dev_path = splits_dir / "dev.jsonl"
    test_path = splits_dir / "test.jsonl"
    ensure_exists(train_path, "Train split")
    ensure_exists(dev_path, "Dev split")
    ensure_exists(test_path, "Test split")
    split_counts = {
        "train": count_rows(train_path),
        "dev": count_rows(dev_path),
        "test": count_rows(test_path),
    }
    if min(split_counts.values()) == 0:
        raise RuntimeError(f"One of the generated splits is empty: {split_counts}")

    run_step(
        [
            sys.executable,
            "scripts/04_train_baselines.py",
            "--train",
            str(train_path),
            "--dev",
            str(dev_path),
            "--test",
            str(test_path),
            "--out",
            str(baselines_dir),
        ]
    )

    run_step(
        [
            sys.executable,
            "scripts/05_build_retrieval_index.py",
            "--train",
            str(train_path),
            "--out",
            str(retrieval_dir),
        ]
    )

    if not args.skip_rag:
        run_step(
            [
                sys.executable,
                "scripts/06_rag_predict.py",
                "--test",
                str(test_path),
                "--index-dir",
                str(retrieval_dir),
                "--out",
                str(rag_out),
                "--model",
                args.rag_model,
                "--limit",
                str(args.rag_limit),
                "--k",
                str(args.rag_k),
            ]
        )

    if args.run_lora:
        run_step(
            [
                sys.executable,
                "scripts/07_finetune_lora.py",
                "--train",
                str(train_path),
                "--dev",
                str(dev_path),
                "--model",
                args.lora_model,
                "--out",
                str(lora_dir),
            ]
        )

    summary = {
        "input": str(input_path),
        "output_root": str(output_root),
        "splits": {
            "train": str(train_path),
            "dev": str(dev_path),
            "test": str(test_path),
        },
        "split_counts": split_counts,
        "baselines_metrics": str(baselines_dir / "metrics.json"),
        "retrieval_dir": str(retrieval_dir),
        "rag_predictions": str(rag_out) if not args.skip_rag else None,
        "lora_dir": str(lora_dir) if args.run_lora else None,
        "config": {
            "seed": args.seed,
            "test_venues": args.test_venues,
            "rag_model": None if args.skip_rag else args.rag_model,
            "rag_limit": None if args.skip_rag else args.rag_limit,
            "rag_k": None if args.skip_rag else args.rag_k,
            "run_lora": args.run_lora,
            "lora_model": args.lora_model if args.run_lora else None,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
