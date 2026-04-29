SYSTEM_PROMPT = """You are an ML conference reviewer assistant. Your task is to predict whether a paper would be accepted based only on pre-review paper content and similar historical papers. Do not use reviewer comments, ratings, rebuttals, or decisions from the target paper."""

RAG_PROMPT = """Given the target paper and similar historical papers, predict acceptance.

Return strict JSON with fields:
- probability_accept: number from 0 to 1
- predicted_label: ACCEPT or REJECT
- rationale: concise explanation
- strengths: list of 2-4 strings
- weaknesses: list of 2-4 strings
- actionable_feedback: list of 3-5 concrete revision suggestions

Similar historical papers:
{contexts}

Target paper:
Title: {title}
Abstract: {abstract}
Text excerpt: {text}
"""
