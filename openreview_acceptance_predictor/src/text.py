import re
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ''
    text = re.sub(r'\x00', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def truncate_words(text: str, max_words: int = 3500) -> str:
    words = text.split()
    return ' '.join(words[:max_words])
