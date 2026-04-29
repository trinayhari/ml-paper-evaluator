import argparse
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_jsonl


def safe_train_test_split(rows, test_size, seed):
    if not rows:
        return [], []
    if len(rows) < 2:
        return rows, []
    labels = [r['label'] for r in rows]
    unique = set(labels)
    n_test = test_size if isinstance(test_size, int) else max(1, int(round(len(rows) * test_size)))
    use_stratify = (
        len(unique) > 1
        and min(labels.count(label) for label in unique) >= 2
        and n_test >= len(unique)
    )
    kwargs = {'test_size': test_size, 'random_state': seed}
    if use_stratify:
        kwargs['stratify'] = labels
    return train_test_split(rows, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--test-venues', nargs='*', default=[])
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    rows = read_jsonl(args.input)
    if args.test_venues:
        test = [r for r in rows if r['venue'] in args.test_venues]
        rem = [r for r in rows if r['venue'] not in args.test_venues]
    else:
        rem, test = safe_train_test_split(rows, test_size=0.15, seed=args.seed)
    train, dev = safe_train_test_split(rem, test_size=0.15, seed=args.seed)
    write_jsonl(train, f'{args.out_dir}/train.jsonl')
    write_jsonl(dev, f'{args.out_dir}/dev.jsonl')
    write_jsonl(test, f'{args.out_dir}/test.jsonl')
    print({'train': len(train), 'dev': len(dev), 'test': len(test)})

if __name__ == '__main__':
    main()
