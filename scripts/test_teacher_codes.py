"""Answer codebook checks, including context-dependent token merging."""
from types import SimpleNamespace
import unittest
from scripts.teacher_codes import aligned, select_codes


class CodeTests(unittest.TestCase):
    def test_prefix_merge_and_failed_round_trip_are_rejected(self):
        class Tokenizer:
            def __init__(self, merge=False, bad_decode=False):
                self.merge, self.bad_decode = merge, bad_decode
            def encode(self, text, add_special_tokens):
                return SimpleNamespace(ids={'Answer:\n': [1, 2], 'Answer:\nAA': [1, 99] if self.merge else [1, 2, 3]}[text])
            def decode(self, ids, skip_special_tokens):
                return 'AB' if self.bad_decode else 'AA'
        self.assertTrue(aligned(Tokenizer(), 'AA', 3, 'Answer:\n'))
        self.assertFalse(aligned(Tokenizer(merge=True), 'AA', 3, 'Answer:\n'))
        self.assertFalse(aligned(Tokenizer(bad_decode=True), 'AA', 3, 'Answer:\n'))

    def test_codebook_selection_is_reproducible_and_bijective(self):
        pool = [{'code': str(i), 'token_id': i} for i in range(300)]
        selected = select_codes(pool, 255, 42)
        self.assertEqual(selected, select_codes(list(reversed(pool)), 255, 42))
        self.assertEqual(len({e['token_id'] for e in selected}), 255)
        self.assertNotEqual(selected, select_codes(pool, 255, 43))

    def test_insufficient_codes_and_duplicate_token_ids_fail(self):
        with self.assertRaises(ValueError): select_codes([], 255, 0)
        with self.assertRaises(ValueError): select_codes([{'code':'AA','token_id':1}, {'code':'AB','token_id':1}], 2, 0)
        with self.assertRaises(ValueError): select_codes([], 256, 0)


if __name__ == '__main__': unittest.main()
