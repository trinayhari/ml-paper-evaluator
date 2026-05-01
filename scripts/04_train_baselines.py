import argparse, json
from pathlib import Path
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl
from src.metrics import classification_report_dict
from src.ml_utils import build_text_fields, train_embedding_logreg


def fields(rows):
    x = build_text_fields(rows, max_words=3500)
    y = [int(r['label']) for r in rows]
    return x, y


def evaluate(name, model, x, y):
    probs = model.predict_proba(x)[:, 1]
    return {name: classification_report_dict(y, probs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--dev', required=True); p.add_argument('--test', required=True); p.add_argument('--out', required=True)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train, dev, test = read_jsonl(args.train), read_jsonl(args.dev), read_jsonl(args.test)
    xtr, ytr = fields(train); xdev, ydev = fields(dev); xte, yte = fields(test)

    tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=80000, ngram_range=(1,2), min_df=1, stop_words='english')),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])
    tfidf.fit(xtr, ytr)
    results = {}
    results.update(evaluate('tfidf_logreg_train', tfidf, xtr, ytr))
    results.update(evaluate('tfidf_logreg_dev', tfidf, xdev, ydev))
    results.update(evaluate('tfidf_logreg_test', tfidf, xte, yte))
    joblib.dump(tfidf, f'{args.out}/tfidf_logreg.joblib')

    embedding_result = train_embedding_logreg(xtr, ytr, xdev, ydev, xte, yte, classification_report_dict)
    model_name = 'sbert_logreg' if embedding_result['backend'] == 'sbert' else 'tfidf_embed_logreg'
    results[f'{model_name}_train'] = embedding_result['train_metrics']
    results[f'{model_name}_dev'] = embedding_result['dev_metrics']
    results[f'{model_name}_test'] = embedding_result['test_metrics']
    joblib.dump(
        {'backend': embedding_result['backend'], 'encoder': embedding_result['encoder'], 'clf': embedding_result['clf']},
        f'{args.out}/{model_name}.joblib'
    )

    with open(f'{args.out}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
