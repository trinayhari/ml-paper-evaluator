import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl
from src.ml_utils import build_retrieval_backend, build_text_fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--out', required=True)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train)
    texts = build_text_fields(rows, max_words=1000)
    backend = build_retrieval_backend(texts, rows, args.out)
    print(f'indexed {len(rows)} papers using {backend}')

if __name__ == '__main__':
    main()
