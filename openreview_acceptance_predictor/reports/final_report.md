# Predicting ML Conference Paper Acceptance from Pre-Review Text

**Team:** Aryan Singh, Trinayaan Hariharan, Saisaketh Koppu, Rishab Kalluri

## 1. Problem and Motivation

Peer review is a slow and noisy feedback loop. Authors often wait months before learning whether a paper was accepted, and the resulting feedback can be inconsistent across reviewers. This project studies whether the paper itself contains enough signal to predict eventual accept/reject outcomes before any reviewer signal is available. The goal is not to replace peer review, but to build an early-stage NLP system that gives authors a calibrated acceptance estimate and concrete revision feedback.

Formally, we model the task as binary classification. Given a paper title, abstract, and full paper text, predict whether the paper is accepted or rejected by a major ML conference. We intentionally exclude reviewer scores, reviewer text, rebuttals, meta-reviews, and discussion threads because those are unavailable at pre-submission time and would create label leakage.

## 2. Data

We collect public submissions and decisions from OpenReview using the official Python API. Each example contains venue, forum ID, title, abstract, PDF text, decision string, and a binary label. We use ICLR and related OpenReview-hosted ML venues as the primary data source. PDFs are downloaded from OpenReview and converted into text with `pypdf`. We remove papers where the extracted text is too short to be reliable.

The final split should be venue/year-aware. For example, we train on older ICLR years and test on a later year. This is more realistic than a random split because it evaluates whether the model generalizes to a future submission pool rather than memorizing year-specific artifacts.

## 3. Methods

### 3.1 Baseline 1: TF-IDF + Logistic Regression

The first baseline represents each paper with TF-IDF unigram and bigram features over the title, abstract, and truncated paper text. A class-balanced logistic regression classifier predicts the probability of acceptance. This baseline is simple, fast, interpretable, and establishes whether surface-level lexical signals are predictive.

### 3.2 Baseline 2: Sentence Embeddings + Logistic Regression

The second baseline uses a SentenceTransformer encoder to embed the same paper text into dense semantic vectors, followed by logistic regression. This tests whether semantic representations improve over sparse lexical features.

### 3.3 Proposed Method: Retrieval-Augmented LLM Prediction

Our main method augments each target paper with the most semantically similar historical papers from the training set. We build a FAISS index over sentence embeddings of historical papers. At inference time, we retrieve the top-k similar accepted/rejected papers and prompt an instruction-tuned open-source LLM to produce strict JSON containing:

- probability of acceptance
- binary prediction
- rationale
- strengths
- weaknesses
- actionable feedback

The intuition is that peer review is inherently comparative. A reviewer does not judge a paper in isolation; they compare novelty, rigor, and empirical strength against related work. Retrieval gives the model local context about what similar accepted and rejected papers looked like historically.

### 3.4 Optional Fine-Tuning

As an extension, we provide a LoRA fine-tuning script for an open-source causal language model. The fine-tuned model learns to map paper text to accept/reject labels and concise rationales. In the final report, the fine-tuned model should only be included if there is enough compute to train and evaluate it properly.

## 4. Evaluation

We evaluate binary classification performance on a held-out year/venue split. Primary metrics are accuracy, precision, recall, F1, and AUROC. Because authors care about whether they can trust the model confidence, we also report Brier score and Expected Calibration Error. Calibration matters because a model that says “70% likely accepted” should be accepted roughly 70% of the time over many examples.

We compare:

1. Majority-class baseline
2. TF-IDF + Logistic Regression
3. SentenceTransformer + Logistic Regression
4. Retrieval-augmented LLM prompting
5. Optional LoRA fine-tuned LLM

A meaningful result would show that the proposed method improves F1/AUROC or calibration over simpler baselines, while also producing more useful explanations.

## 5. Expected Findings and Error Analysis

We expect the task to be difficult because acceptance depends on reviewer preferences, conference capacity, novelty relative to unpublished work, and stochastic review assignment. However, paper text should still contain signals such as clarity, experimental rigor, breadth of evaluation, writing quality, and positioning against related work.

Error analysis should categorize false positives and false negatives. False positives may include well-written but incremental papers. False negatives may include technically strong papers whose contribution is subtle or highly specialized. We also inspect whether the model is over-relying on superficial cues such as paper length, benchmark names, or popular domains.

## 6. Conclusion

This project builds a pre-review acceptance prediction system using OpenReview paper text. The key contribution is a leakage-free framing: predictions are made from paper content alone, before reviewer scores or discussion signals exist. The system combines classical NLP baselines, dense retrieval, LLM-based structured prediction, and calibration analysis. Beyond binary classification, the model produces actionable feedback, making it useful as an author-facing writing aid rather than only a benchmark classifier.
