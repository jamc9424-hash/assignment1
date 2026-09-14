import gzip
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = Path('C:/Users/mcken/AppData/Local/hermes/outputs/gift-cards-check/Gift_Cards.jsonl.gz')
OUT = ROOT / 'evaluation' / 'balanced-3class-150'
OUT.mkdir(parents=True, exist_ok=True)
seed = 6418
per_class = 50
buckets = {'POSITIVE': [], 'NEUTRAL': [], 'NEGATIVE': []}
with gzip.open(SOURCE, 'rt', encoding='utf-8') as f:
    for line_number, line in enumerate(f, 1):
        r = json.loads(line)
        rating = r['rating']
        label = 'POSITIVE' if rating >= 4 else 'NEUTRAL' if rating == 3 else 'NEGATIVE'
        buckets[label].append({'source_line': line_number, 'rating': rating, 'title': r['title'], 'text': r['text']})
rng = random.Random(seed)
sampled = []
for label in ('POSITIVE', 'NEUTRAL', 'NEGATIVE'):
    assert len(buckets[label]) >= per_class
    chosen = rng.sample(buckets[label], per_class)
    for item in chosen:
        sampled.append({'id': len(sampled) + 1, 'source_line': item['source_line'], 'rating': item['rating'], 'reference': label, 'title': item['title'], 'text': item['text']})
rng.shuffle(sampled)
for i, item in enumerate(sampled, 1):
    item['id'] = i
reference = [{'id': x['id'], 'source_line': x['source_line'], 'rating': x['rating'], 'reference': x['reference']} for x in sampled]
(OUT / 'reference_labels.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
with open(OUT / 'balanced_inputs.jsonl', 'w', encoding='utf-8') as f:
    for x in sampled:
        f.write(json.dumps({'id': x['id'], 'title': x['title'], 'text': x['text']}, ensure_ascii=False) + '\n')
for batch in range(3):
    rows = sampled[batch * 50:(batch + 1) * 50]
    (OUT / f'blind_batch_{batch+1}.json').write_text(json.dumps([{'id': x['id'], 'title': x['title'], 'text': x['text']} for x in rows], ensure_ascii=False, indent=2), encoding='utf-8')
manifest = {'source_file': str(SOURCE), 'source_reviews_scanned': sum(map(len, buckets.values())), 'seed': seed, 'sample_per_class': per_class, 'total': len(sampled), 'class_rule': {'POSITIVE': 'rating 4-5', 'NEUTRAL': 'rating 3', 'NEGATIVE': 'rating 1-2'}, 'inputs_exclude': ['rating', 'reference', 'source_line'], 'class_counts': {k: sum(x['reference'] == k for x in sampled) for k in buckets}}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
