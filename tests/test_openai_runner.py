import json
import unittest

from run_openai_compatible import build_request, parse_chat_completion


class OpenAICompatibleTests(unittest.TestCase):
    def test_build_request_uses_blind_review_messages(self):
        request = build_request('demo-model', [{'role': 'user', 'content': '{}'}])
        self.assertEqual(request['model'], 'demo-model')
        self.assertEqual(request['messages'][0]['role'], 'user')
        self.assertEqual(request['temperature'], 0)

    def test_parse_chat_completion_returns_message_content(self):
        payload = {'choices': [{'message': {'content': '{"sentiment":"NEUTRAL","emotion":"trust"}'}}]}
        self.assertEqual(parse_chat_completion(payload), payload['choices'][0]['message']['content'])

    def test_parse_chat_completion_rejects_missing_content(self):
        with self.assertRaises(ValueError):
            parse_chat_completion({'choices': []})


if __name__ == '__main__':
    unittest.main()
