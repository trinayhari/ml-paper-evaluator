import argparse
from pathlib import Path
import sys
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_jsonl
from src.text import normalize_text


def pdf_url(pdf_field: str) -> str:
    if not pdf_field:
        return ''
    if pdf_field.startswith('http'):
        return pdf_field
    return 'https://openreview.net' + pdf_field


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    path.write_bytes(r.content)


def extract_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError('pypdf is required to extract PDF text') from exc
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or '')
        except Exception:
            continue
    return normalize_text('\n'.join(pages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--pdf-dir', required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    out_rows = []
    for row in tqdm(rows):
        url = pdf_url(row.get('pdf', ''))
        if not url:
            continue
        pdf_path = Path(args.pdf_dir) / f"{row['forum']}.pdf"
        try:
            download(url, pdf_path)
            text = extract_text(pdf_path)
            if len(text.split()) < 500:
                continue
            row['pdf_path'] = str(pdf_path)
            row['paper_text'] = text
            out_rows.append(row)
        except Exception as e:
            print(f"WARN: failed {row.get('forum')}: {e}")
    write_jsonl(out_rows, args.out)
    print(f'wrote {len(out_rows)} papers to {args.out}')


if __name__ == '__main__':
    main()
