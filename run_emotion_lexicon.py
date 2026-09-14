import csv
import hashlib
import json
from pathlib import Path

from emotion_lexicon import load_nrc_lexicon, score_primary_emotion

ROOT = Path(__file__).parent
EVAL = ROOT / 'evaluation' / 'gift-cards-batch-100'
with (EVAL / 'scored_reviews.csv').open(encoding='utf-8-sig', newline='') as f:
    reviews = list(csv.DictReader(f))
assert len(reviews) == 100
lex_path = ROOT / 'data' / 'NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'
lexicon = load_nrc_lexicon(lex_path.read_text(encoding='utf-8'))
assert len(lexicon) > 4000
for batch in range(4):
    rows = reviews[batch * 25:(batch + 1) * 25]
    blind = [{'id': int(r['id']), 'title': r['title'], 'text': r['text']} for r in rows]
    (EVAL / f'emotion_blind_batch_{batch + 1}.json').write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding='utf-8')
lex_rows = []
for r in reviews:
    result = score_primary_emotion(r['title'], r['text'], lexicon)
    lex_rows.append({'id': int(r['id']), 'lexicon_emotion': result['primary_emotion'], 'lexicon_scores': result['scores'], 'matched_words': result['matched_words']})
(EVAL / 'lexicon_emotions.json').write_text(json.dumps(lex_rows, ensure_ascii=False, indent=2), encoding='utf-8')
manifest = {'n': 100, 'source': 'evaluation/gift-cards-batch-100/scored_reviews.csv', 'lexicon_file': str(lex_path.relative_to(ROOT)), 'lexicon_sha256': hashlib.sha256(lex_path.read_bytes()).hexdigest(), 'lexicon_loader': 'emotion_lexicon.load_nrc_lexicon', 'tokenizer': r"[A-Za-z]+(?:['-][A-Za-z]+)?", 'counting': 'Each matching title/text token adds one point to every associated NRC emotion; ties use EMOTIONS order; zero matches returns null.', 'llm_inputs': 'Four files contain only id, title, text; ratings and rating-derived labels excluded.', 'emotions': ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']}
(EVAL / 'emotion_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Prepared {len(reviews)} blinded emotion inputs and scored {len(lex_rows)} reviews with {len(lexicon)} lexicon entries.')
