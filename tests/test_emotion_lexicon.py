import unittest


class LexiconTests(unittest.TestCase):
    def test_scores_words_and_uses_deterministic_tie_order(self):
        from emotion_lexicon import score_primary_emotion
        lexicon = {
            'happy': {'joy'}, 'delighted': {'joy'}, 'safe': {'trust'},
            'angry': {'anger'}, 'worried': {'fear'},
        }
        result = score_primary_emotion('Happy card', 'I feel happy and safe.', lexicon)
        self.assertEqual(result['primary_emotion'], 'joy')
        self.assertEqual(result['scores']['joy'], 2)
        self.assertEqual(result['scores']['trust'], 1)
        tie = score_primary_emotion('angry', 'worried', lexicon)
        self.assertEqual(tie['primary_emotion'], 'anger')

    def test_reports_no_match_without_inventing_emotion(self):
        from emotion_lexicon import score_primary_emotion
        result = score_primary_emotion('ASIN B123', 'qzxv plmokn', {})
        self.assertIsNone(result['primary_emotion'])
        self.assertEqual(sum(result['scores'].values()), 0)

    def test_loads_nrc_tsv_associations(self):
        from emotion_lexicon import load_nrc_lexicon
        result = load_nrc_lexicon('happy\tjoy\t1\nnotjoy\tjoy\t0\n')
        self.assertEqual(result, {'happy': {'joy'}})


if __name__ == '__main__':
    unittest.main()
