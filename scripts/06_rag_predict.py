import argparse, json, re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl, write_jsonl
from src.text import truncate_words
from src.prompts import SYSTEM_PROMPT, RAG_PROMPT
from src.ml_utils import load_retrieval_backend, query_retrieval


def make_context(rows, idxs):
    chunks = []
    for i in idxs:
        r = rows[int(i)]
        label = 'ACCEPT' if int(r['label']) == 1 else 'REJECT'
        chunks.append(f"Title: {r.get('title','')}\nOutcome: {label}\nAbstract: {truncate_words(r.get('abstract',''), 160)}")
    return '\n\n'.join(chunks)


def parse_prob(text):
    m = re.search(r'probability_accept"?\s*:?\s*([0-9.]+)', text)
    if not m:
        return 0.5
    return max(0.0, min(1.0, float(m.group(1))))


def heuristic_prediction(row, neighbors):
    if not neighbors:
        prob = 0.5
    else:
        prob = sum(int(n['label']) for n in neighbors) / len(neighbors)
    label = 'ACCEPT' if prob >= 0.5 else 'REJECT'
    return {
        'probability_accept': round(prob, 4),
        'predicted_label': label,
        'rationale': 'Fallback heuristic based on labels of nearest historical papers.',
        'strengths': [f"Retrieved {len(neighbors)} similar historical papers."],
        'weaknesses': ['LLM generation was unavailable, so this output is retrieval-only.'],
        'actionable_feedback': ['Run again with a local or downloadable text-generation model for richer feedback.'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True); p.add_argument('--index-dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2')
    p.add_argument('--limit', type=int, default=None); p.add_argument('--k', type=int, default=5)
    args = p.parse_args()
    test_rows = read_jsonl(args.test)[:args.limit]
    train_rows = read_jsonl(f'{args.index_dir}/rows.jsonl')
    retrieval = load_retrieval_backend(args.index_dir)
    gen = None
    try:
        from transformers import pipeline

        gen = pipeline('text-generation', model=args.model, device_map='auto', max_new_tokens=512)
    except Exception as exc:
        print(f'WARN: text generation disabled, using retrieval-only fallback: {exc}')
    outs = []
    for r in test_rows:
        query = truncate_words(r.get('title','') + '\n' + r.get('abstract','') + '\n' + r.get('paper_text',''), 1000)
        idxs = query_retrieval(retrieval, query, args.k)
        neighbors = [train_rows[int(i)] for i in idxs]
        prediction_payload = heuristic_prediction(r, neighbors)
        pred = json.dumps(prediction_payload)
        if gen is not None:
            prompt = RAG_PROMPT.format(contexts=make_context(train_rows, idxs), title=r.get('title',''), abstract=r.get('abstract',''), text=truncate_words(r.get('paper_text',''), 1800))
            full_prompt = f'<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]'
            pred = gen(full_prompt)[0]['generated_text']
        outs.append({'forum': r['forum'], 'true_label': r['label'], 'prediction_text': pred, 'probability_accept': parse_prob(pred)})
    write_jsonl(outs, args.out)
    print(f'wrote {len(outs)} predictions to {args.out}')

if __name__ == '__main__':
    main()
