from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def build_text_fields(rows, max_words: int) -> list[str]:
    texts = []
    for row in rows:
        text = "\n".join(
            [row.get("title", ""), row.get("abstract", ""), row.get("paper_text", "")]
        )
        texts.append(" ".join(text.split()[:max_words]))
    return texts


def maybe_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        print(f"WARN: falling back to TF-IDF because SentenceTransformer('{model_name}') failed: {exc}")
        return None


def encode_texts(texts, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedder = maybe_sentence_transformer(model_name)
    if embedder is not None:
        matrix = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return "sbert", embedder, np.asarray(matrix, dtype="float32")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    return "tfidf", vectorizer, matrix


def transform_texts(backend, encoder, texts):
    if backend == "sbert":
        matrix = encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype="float32")
    return encoder.transform(texts)


def train_embedding_logreg(xtr, ytr, xdev, ydev, xte, yte, metrics_fn):
    backend, encoder, ztr = encode_texts(xtr)
    zdev = transform_texts(backend, encoder, xdev)
    zte = transform_texts(backend, encoder, xte)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(ztr, ytr)
    return {
        "backend": backend,
        "encoder": encoder,
        "clf": clf,
        "train_metrics": metrics_fn(ytr, clf.predict_proba(ztr)[:, 1]),
        "dev_metrics": metrics_fn(ydev, clf.predict_proba(zdev)[:, 1]),
        "test_metrics": metrics_fn(yte, clf.predict_proba(zte)[:, 1]),
    }


def build_retrieval_backend(texts, rows, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    backend, encoder, matrix = encode_texts(texts, model_name=model_name)

    with open(out_path / "rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    if backend == "sbert":
        try:
            import faiss
        except ImportError:
            backend = "tfidf"
            vectorizer = TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
            )
            matrix = vectorizer.fit_transform(texts)
            encoder = vectorizer
        else:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(out_path / "index.faiss"))
            joblib.dump({"backend": backend, "model_name": model_name}, out_path / "index_meta.joblib")
            return backend

    nn = NearestNeighbors(metric="cosine")
    nn.fit(matrix)
    joblib.dump(
        {"backend": backend, "encoder": encoder, "matrix": matrix, "nn": nn},
        out_path / "index.joblib",
    )
    return backend


def load_retrieval_backend(index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    index_path = Path(index_dir)
    faiss_path = index_path / "index.faiss"
    meta_path = index_path / "index_meta.joblib"
    if faiss_path.exists() and meta_path.exists():
        meta = joblib.load(meta_path)
        import faiss

        return {
            "backend": meta["backend"],
            "encoder": maybe_sentence_transformer(model_name),
            "index": faiss.read_index(str(faiss_path)),
        }
    return joblib.load(index_path / "index.joblib")


def query_retrieval(retrieval, query_text: str, k: int):
    k = max(1, min(k, retrieval["matrix"].shape[0] if "matrix" in retrieval else retrieval["index"].ntotal))
    backend = retrieval["backend"]
    if backend == "sbert":
        query = transform_texts(backend, retrieval["encoder"], [query_text])
        _, idxs = retrieval["index"].search(query, k)
        return idxs[0].tolist()

    query = retrieval["encoder"].transform([query_text])
    _, idxs = retrieval["nn"].kneighbors(query, n_neighbors=k)
    return idxs[0].tolist()
