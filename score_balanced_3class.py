import csv
import json
from collections import Counter
from pathlib import Path

from emotion_lexicon import load_nrc_lexicon, score_primary_emotion
from sentiment_prompt import parse_sentiment_emotion

ROOT = Path(__file__).parent
EVAL = ROOT / 'evaluation' / 'balanced-3class-150'
llm = sum((json.loads((EVAL / f'llm_batch_{b}.json').read_text(encoding='utf-8')) for b in range(1, 4)), [])
assert len(llm) == 150 and {x['id'] for x in llm} == set(range(1,151))
for x in llm:
    parse_sentiment_emotion(json.dumps({'sentiment': x['sentiment'], 'emotion': x['emotion']}))
llm = {x['id']: x for x in llm}
refs = {x['id']: x for x in json.loads((EVAL / 'reference_labels.json').read_text(encoding='utf-8'))}
assert len(refs) == 150 and set(refs) == set(llm)
with (EVAL / 'balanced_inputs.jsonl').open(encoding='utf-8') as f:
    inputs = {x['id']: x for x in map(json.loads, f)}
lex_path = ROOT / 'data' / 'NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'
lexicon = load_nrc_lexicon(lex_path.read_text(encoding='utf-8'))
rows=[]
for i in range(1,151):
    x=inputs[i]; l=llm[i]; ref=refs[i]
    lex=score_primary_emotion(x['title'],x['text'],lexicon)
    rows.append({'id':i,'source_line':ref['source_line'],'rating':float(ref['rating']),'reference':ref['reference'],'prediction':l['sentiment'],'correct':l['sentiment']==ref['reference'],'llm_emotion':l['emotion'],'lexicon_emotion':lex['primary_emotion'],'lexicon_scores':lex['scores'],'matched_words':lex['matched_words'],'emotion_agreement':lex['primary_emotion']==l['emotion'] if lex['primary_emotion'] else None,'title':x['title'],'text':x['text']})
labels=('POSITIVE','NEUTRAL','NEGATIVE')
cm=Counter((r['reference'],r['prediction']) for r in rows)
per={}
for lab in labels:
 tp=cm[lab,lab]; support=sum(r['reference']==lab for r in rows); pred=sum(r['prediction']==lab for r in rows); precision=tp/pred if pred else 0; recall=tp/support if support else 0; per[lab]={'support':support,'predicted':pred,'precision':round(precision,4),'recall':round(recall,4),'f1':round(2*precision*recall/(precision+recall),4) if precision+recall else 0}
correct=sum(r['correct'] for r in rows); compared=[r for r in rows if r['lexicon_emotion']]
agr=sum(r['emotion_agreement'] for r in compared); llm_dist=Counter(r['llm_emotion'] for r in rows); lex_dist=Counter(r['lexicon_emotion'] or 'no_match' for r in rows)
emotion_cm=Counter((r['llm_emotion'],r['lexicon_emotion']) for r in compared)
report={'n':150,'correct':correct,'incorrect':150-correct,'accuracy':round(correct/150,4),'macro_f1':round(sum(x['f1'] for x in per.values())/3,4),'balanced_accuracy':round(sum(x['recall'] for x in per.values())/3,4),'confusion_matrix':{a:{b:cm[a,b] for b in labels} for a in labels},'per_class':per,'reference_distribution':dict(Counter(r['reference'] for r in rows)),'prediction_distribution':dict(Counter(r['prediction'] for r in rows)),'three_star_n':sum(r['rating']==3 for r in rows),'three_star_predictions':dict(Counter(r['prediction'] for r in rows if r['rating']==3)),'three_star_correct':sum(r['rating']==3 and r['correct'] for r in rows),'emotion_compared_n':len(compared),'emotion_agreements':agr,'emotion_disagreements':len(compared)-agr,'emotion_agreement_rate':round(agr/len(compared),4) if compared else None,'emotion_all_rows_agreement_rate':round(agr/150,4),'lexicon_no_match':sum(not r['lexicon_emotion'] for r in rows),'llm_emotion_distribution':dict(llm_dist),'lexicon_emotion_distribution':dict(lex_dist),'emotion_confusion_matrix':{a:{b:emotion_cm[a,b] for b in sorted(set(x['llm_emotion'] for x in rows)|set(x['lexicon_emotion'] for x in rows if x['lexicon_emotion']))} for a in sorted(set(x['llm_emotion'] for x in rows))},'errors':[r for r in rows if not r['correct']]}
(EVAL/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
(EVAL/'scored_reviews.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
with (EVAL/'scored_reviews.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['id','source_line','rating','reference','prediction','correct','llm_emotion','lexicon_emotion','emotion_agreement','title','text','matched_words']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        q={k:r[k] for k in fields}; q['matched_words']=' '.join(r['matched_words']); w.writerow(q)
print(json.dumps({k:report[k] for k in ('n','correct','incorrect','accuracy','macro_f1','balanced_accuracy','confusion_matrix','per_class','reference_distribution','prediction_distribution','three_star_predictions','three_star_correct','emotion_compared_n','emotion_agreements','emotion_disagreements','emotion_agreement_rate','emotion_all_rows_agreement_rate','lexicon_no_match','llm_emotion_distribution','lexicon_emotion_distribution')},indent=2))
