import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
EVAL = ROOT / 'evaluation' / 'gift-cards-batch-100'
llm = []
for batch in range(1, 5):
    rows = json.loads((EVAL / f'llm_emotions_{batch}.json').read_text(encoding='utf-8'))
    assert len(rows) == 25
    llm.extend(rows)
assert len(llm) == 100 and {r['id'] for r in llm} == set(range(1, 101))
emotions = {'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'}
assert all(set(r) == {'id', 'sentiment', 'emotion'} and r['emotion'] in emotions for r in llm)
llm_by_id = {r['id']: r for r in llm}
lex_rows = json.loads((EVAL / 'lexicon_emotions.json').read_text(encoding='utf-8'))
assert len(lex_rows) == 100 and {r['id'] for r in lex_rows} == set(range(1, 101))
with (EVAL / 'scored_reviews.csv').open(encoding='utf-8-sig', newline='') as f:
    reviews = {int(r['id']): r for r in csv.DictReader(f)}
joined = []
for lex in sorted(lex_rows, key=lambda x: x['id']):
    i = lex['id']; l = llm_by_id[i]; r = reviews[i]
    joined.append({'id': i, 'title': r['title'], 'rating': float(r['rating']), 'sentiment': l['sentiment'], 'llm_emotion': l['emotion'], 'lexicon_emotion': lex['lexicon_emotion'], 'lexicon_scores': lex['lexicon_scores'], 'matched_words': lex['matched_words'], 'emotion_agreement': lex['lexicon_emotion'] == l['emotion'] if lex['lexicon_emotion'] else None})
compared = [r for r in joined if r['lexicon_emotion'] is not None]
agree = sum(r['emotion_agreement'] for r in compared)
conf = Counter((r['llm_emotion'], r['lexicon_emotion']) for r in compared)
llm_dist = Counter(r['llm_emotion'] for r in joined); lex_dist = Counter(r['lexicon_emotion'] or 'no_match' for r in joined)
by_emotion = {}
for e in sorted(emotions):
    n = sum(r['llm_emotion'] == e for r in compared); same = sum(r['llm_emotion'] == e and r['emotion_agreement'] for r in compared)
    by_emotion[e] = {'llm_count': llm_dist[e], 'lexicon_count': lex_dist[e], 'compared_count': n, 'agreements': same, 'agreement_rate': round(same/n, 4) if n else None}
report = {'n': 100, 'llm_predictions': len(llm), 'lexicon_predictions': len(lex_rows), 'lexicon_no_match': sum(r['lexicon_emotion'] is None for r in joined), 'compared_n': len(compared), 'agreement_count': agree, 'disagreement_count': len(compared)-agree, 'agreement_rate': round(agree/len(compared), 4) if compared else None, 'agreement_rate_all_rows_treating_no_match_as_disagreement': round(agree/100, 4), 'llm_distribution': dict(llm_dist), 'lexicon_distribution_including_no_match': dict(lex_dist), 'confusion_matrix_llm_rows_lexicon_columns': {a: {b: conf[a,b] for b in sorted(emotions)} for a in sorted(emotions)}, 'by_emotion': by_emotion, 'disagreements': [r for r in joined if r['emotion_agreement'] is False], 'no_match_examples': [r for r in joined if r['lexicon_emotion'] is None][:20]}
(EVAL / 'emotion_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
with (EVAL / 'emotion_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
    fields = ['id', 'title', 'rating', 'sentiment', 'llm_emotion', 'lexicon_emotion', 'emotion_agreement', 'lexicon_scores', 'matched_words']
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in joined:
        row = dict(r); row['lexicon_scores'] = json.dumps(row['lexicon_scores'], sort_keys=True); row['matched_words'] = ' '.join(row['matched_words']); w.writerow(row)
print(json.dumps({k: report[k] for k in ('n','compared_n','agreement_count','disagreement_count','agreement_rate','agreement_rate_all_rows_treating_no_match_as_disagreement','lexicon_no_match','llm_distribution','lexicon_distribution_including_no_match','by_emotion')}, indent=2))
