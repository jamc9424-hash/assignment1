"""Run blinded review inputs through an OpenAI-compatible chat endpoint.

Usage:
  OPENAI_API_KEY=... OPENAI_MODEL=... python run_openai_compatible.py \
    evaluation/balanced-3class-150/blind_batch_1.json \
    evaluation/balanced-3class-150/endpoint_batch_1.json

The input must contain only id, title, and text. The output contains id,
sentiment, and emotion; ratings and reference labels are never sent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sentiment_prompt import build_sentiment_emotion_messages, parse_sentiment_emotion


def build_request(model: str, messages: list[dict[str, str]]) -> dict:
    """Build a deterministic Chat Completions request."""
    return {
        'model': model,
        'messages': messages,
        'temperature': 0,
    }


def parse_chat_completion(payload: dict) -> str:
    """Extract assistant text from a Chat Completions JSON response."""
    try:
        content = payload['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Chat completion did not contain message content') from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Chat completion message content must be non-empty text')
    return content


def call_endpoint(base_url: str, api_key: str, model: str,
                  messages: list[dict[str, str]], timeout: float) -> dict[str, str]:
    """Call /chat/completions and validate its strict JSON answer."""
    url = base_url.rstrip('/') + '/chat/completions'
    body = json.dumps(build_request(model, messages), ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:500]
        raise RuntimeError(f'Endpoint returned HTTP {exc.code}: {detail}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Endpoint request failed: {exc.reason}') from exc
    return parse_sentiment_emotion(parse_chat_completion(payload))


def load_blind_rows(path: Path) -> list[dict]:
    """Load and enforce the blind input schema."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        row = json.loads(line)
        if set(row) != {'id', 'title', 'text'}:
            raise ValueError(f'{path}:{line_number} is not a blind id/title/text row')
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSONL file containing id, title, text')
    parser.add_argument('output', type=Path, help='JSON file for validated predictions')
    parser.add_argument('--base-url', default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--model', default=os.getenv('OPENAI_MODEL'))
    parser.add_argument('--timeout', type=float, default=120.0)
    args = parser.parse_args(argv)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        parser.error('OPENAI_API_KEY is required')
    if not args.model:
        parser.error('OPENAI_MODEL or --model is required')

    outputs = []
    for row in load_blind_rows(args.input):
        prediction = call_endpoint(
            args.base_url, api_key, args.model,
            build_sentiment_emotion_messages(row['title'], row['text']), args.timeout,
        )
        outputs.append({'id': row['id'], **prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output} ({len(outputs)} predictions)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
