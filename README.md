# Amazon Gift Cards sentiment classification

MBAX6418 assignment 1. Python 3.10+; standard library only.

## Current feature: reusable three-class sentiment prompt

`sentiment_prompt.py` provides:
- `SYSTEM_PROMPT`: versioned instructions for `POSITIVE`, `NEUTRAL`, or `NEGATIVE`.
- `build_sentiment_messages(title, text)`: role-separated messages containing only
  the review title and text.
- `build_sentiment_emotion_messages(title, text)`: strict JSON prompt for sentiment
  plus one primary NRC emotion.
- `parse_sentiment` and `parse_sentiment_emotion`: strict validators that reject
  malformed or out-of-vocabulary model output rather than guessing.

The current rating reference convention is **4–5 = POSITIVE, 3 = NEUTRAL, and
1–2 = NEGATIVE**. The prompt treats neutral/mixed reviews as NEUTRAL when no
assessment dominates, while preserving guidance for sarcasm, passive aggression,
negation, conflicting title/body, and unresolved purchase failures. Ratings are
never sent to the LLM.

This feature constructs prompts; it does not make an API request. Astra is the
intended LLM, but no endpoint, credentials, SDK, or provider model ID is assumed.

### Manual examples (synthetic; not measured results)

| Title | Text | Expected label |
|---|---|---|
| Thanks for nothing | A card with zero balance. | NEGATIVE |
| Works fine | It arrived and redeemed without trouble. | POSITIVE |
| Gift card | Delivered Tuesday; it is a twenty-dollar card. | NEUTRAL |
| Good gift | The envelope was bent, but the card worked and Dad loved it. | POSITIVE |
| Easy to use | Very easy to use, though I wish I knew earlier. | NEUTRAL |

## Evaluation: first 100 reviews

The blinded first-batch evaluation is preserved under `evaluation/gift-cards-batch-100/`.
Astra classified each review from title and text only; ratings were read afterward
and converted to the reference rule `4–5 stars = POSITIVE`, `1–3 stars = NEGATIVE`.
Results: 98/100 accuracy, 92.31% macro F1, and 92.31% balanced accuracy. The HTML
file provides the visual score-vs-rating report. These are rating-proxy results,
not gold sentiment labels; see `metrics.json` for the two disagreements and full
confusion matrix. `manifest.json` records the prompt/source hashes and leakage
controls. Reviewer text is included in `scored_reviews.csv`; do not add reviewer
IDs or the raw dataset.

## Emotion comparison: LLM vs NRC lexicon

Step 5 keeps two independent primary-emotion outputs for the 100-review batch:

- `build_sentiment_emotion_messages` asks Astra for strict JSON containing sentiment
  and one of eight NRC emotions.
- `emotion_lexicon.score_primary_emotion` tokenizes title + text, adds one point per
  matching NRC emotion association, uses a fixed order for ties, and returns `null`
  when no emotion-bearing word matches.

The NRC comparison found 19 agreements among 85 reviews with a lexicon match
(**22.35%**), plus 15 no-match reviews. Treating no-match as non-agreement gives
19/100 (**19%**). NRC selected anticipation for 59 reviews, while Astra selected joy
for 48 and trust for 42. This is expected to diverge: word counts are sensitive to
surface words, while the LLM uses context. Neither output is gold emotion truth.
Open `evaluation/gift-cards-batch-100/emotion-comparison.html` for the visual
comparison and evidence table. `emotion_comparison.json` and `.csv` contain all
joinable results and matched-word evidence.

Source: NRC Emotion Lexicon v0.92 from the [public repository](https://github.com/Franck-Dernoncourt/NRC_Emotion_Lexicon).
The NRC source README states that commercial use requires permission from NRC; see
`data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt` for the preserved source.
## Balanced three-class benchmark

The current benchmark samples 50 reviews per class from the entire 152,410-review
file with fixed seed 6418. It is stored under `evaluation/balanced-3class-150/`.
The results are 113/150 correct (**75.33%**), macro F1 **72.08%**, and balanced
accuracy **75.33%**. Per-class recall is POSITIVE **98%**, NEUTRAL **34%**, and
NEGATIVE **94%**. Among the 50 three-star reviews, Astra predicted NEUTRAL 17,
NEGATIVE 25, and POSITIVE 8: the model gives 3-star reviews their own class but
still collapses most of them into polarity, especially negative.

Open `evaluation/balanced-3class-150/dashboard.html` for the confusion matrix,
per-class metrics, 3-star routing view, and review-level audit. The fixed sample
and post-hoc labels are in `manifest.json` and `reference_labels.json`; blind
inputs contain no ratings or reference labels. `metrics.json` contains the exact
scoring output.
## Run tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same suite on pushes and pull requests. No API credentials
are needed for tests. Never commit tokens or the raw review dataset.

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), Gift Cards category,
collected by the McAuley Lab at UC San Diego.
[Download the gzip JSON Lines file](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz).

The preceding full-file verification parsed 152,410 records with all ten expected
fields, found 49 blank review bodies, and measured 84.15% five-star ratings. This
rating imbalance is not a measured sentiment-label distribution. The data contains
ratings, not gold sentiment or emotion labels. Preserve raw metadata for future
analysis and grouping, but keep it out of the sentiment prompt.
