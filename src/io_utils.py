import json
import gzip
from pathlib import Path
from typing import Dict, Iterable, List


def _clean_surrogates(value):
    if isinstance(value, str):
        return value.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    if isinstance(value, list):
        return [_clean_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_surrogates(item) for key, item in value.items()}
    return value


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(_clean_surrogates(row), ensure_ascii=False) + '\n')
