"""Dependency-free NRC emotion lexicon scoring."""

import re
from collections import Counter

from sentiment_prompt import EMOTIONS

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?")


def load_nrc_lexicon(tsv_text: str) -> dict[str, set[str]]:
    """Load NRC word-emotion associations from its tab-separated format."""
    lexicon: dict[str, set[str]] = {}
    for line in tsv_text.splitlines():
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        word, category, flag = parts
        if flag == '1' and category in EMOTIONS:
            lexicon.setdefault(word.lower(), set()).add(category)
    return lexicon


def score_primary_emotion(title: str, text: str, lexicon: dict[str, set[str]]) -> dict:
    """Score title and text; ties follow EMOTIONS order, no-match returns None."""
    if not isinstance(title, str) or not isinstance(text, str):
        raise TypeError('title and text must both be strings')
    scores = Counter({emotion: 0 for emotion in EMOTIONS})
    matched_words = []
    for token in TOKEN_RE.findall(f'{title} {text}'.lower()):
        emotions = lexicon.get(token, set())
        if emotions:
            matched_words.append(token)
            for emotion in emotions:
                scores[emotion] += 1
    maximum = max(scores.values(), default=0)
    primary = next((emotion for emotion in EMOTIONS if scores[emotion] == maximum), None) if maximum else None
    return {'primary_emotion': primary, 'scores': dict(scores), 'matched_words': matched_words}
