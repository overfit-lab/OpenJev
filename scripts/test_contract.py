"""Public contract and input compilation properties."""
from copy import deepcopy
import math
import unittest
from scripts.contract import (BOUNDARIES, compile_request, validate_request, response_from_logits)

class Tokenizer:
    unk_token_id = -1
    def convert_tokens_to_ids(self, items): return [1000 + BOUNDARIES.index(x) for x in items]
    def encode(self, text, add_special_tokens=False): return [ord(x) + 2000 for x in text]

class ContractTest(unittest.TestCase):
    def request(self):
        return {'model': 'jev-latest', 'state': {'description': 'A PostgreSQL sponsor for October'},
                'questions': {
                    'inquiry': {'type': 'noul', 'instructions': 'Is this a sponsor inquiry?'},
                    'category': {'type': 'choice', 'instructions': 'What product?',
                                 'criteria': {'dev_tool': 'Developer hosting', 'course': 'Training', 'other': None}},
                    'quality': {'type': 'score', 'instructions': 'How specific?',
                                'criteria': ['Generic', {'ask': 'Concrete'}, ['Timeframe', 'Product']]}}}
    def test_question_ids_order_and_model_do_not_enter_tokens(self):
        original = self.request(); changed = deepcopy(original)
        changed['model'] = 'openjev-preview'
        changed['questions']['renamed'] = changed['questions'].pop('category')
        changed['questions']['renamed']['criteria'] = dict(reversed(list(changed['questions']['renamed']['criteria'].items())))
        a = {q.kind: q.paths for q in compile_request(Tokenizer(), original)}
        b = {q.kind: q.paths for q in compile_request(Tokenizer(), changed)}
        self.assertEqual(a, b)
    def test_semantic_name_is_visible_even_with_null_description(self):
        a = self.request(); b = deepcopy(a)
        b['questions']['category']['criteria']['unknown'] = b['questions']['category']['criteria'].pop('other')
        self.assertNotEqual(compile_request(Tokenizer(), a)[1].paths, compile_request(Tokenizer(), b)[1].paths)
    def test_score_levels_independent_of_neighbors_and_index(self):
        a = self.request(); b = deepcopy(a)
        b['questions']['quality']['criteria'].reverse()
        self.assertEqual(compile_request(Tokenizer(), a)[2].paths, list(reversed(compile_request(Tokenizer(), b)[2].paths)))
    def test_255_options_and_strict_limits(self):
        r = self.request()
        r['questions'] = {'q': {'type': 'choice', 'instructions': 'Pick', 'criteria': {str(i): None for i in range(255)}}}
        self.assertEqual(len(compile_request(Tokenizer(), r, max_length=20000, max_total_tokens=3000000)[0].paths), 255)
        r['questions']['q']['criteria']['extra'] = None
        with self.assertRaises(ValueError): validate_request(r)
        r['questions']['q']['criteria'] = {'only': None}
        with self.assertRaises(ValueError): validate_request(r)
    def test_no_truncation_and_no_control_token_injection(self):
        r = self.request(); r['state'] = BOUNDARIES[-1]
        for q in compile_request(Tokenizer(), r):
            for path in q.paths: self.assertEqual(path.count(1003), 1)
        with self.assertRaisesRegex(ValueError, 'no truncation'): compile_request(Tokenizer(), r, max_length=5)
        with self.assertRaisesRegex(ValueError, 'total'): compile_request(Tokenizer(), r, max_total_tokens=5)
    def test_response_has_exact_public_shapes(self):
        r = self.request(); z = {'inquiry': [0., 3.], 'category': [0., 4., 1.], 'quality': [0., 1., 2.]}
        answer = response_from_logits(r, z, 'openjev-test', tokenizer=Tokenizer(), input_tokens=210)
        self.assertEqual(set(answer), {'model', 'answers', 'usage'})
        expected = {'inquiry': {'type','noul'}, 'category': {'type','choice','probabilities','confidence'},
                    'quality': {'type','score','legend','probabilities','confidence'}}
        for key, fields in expected.items(): self.assertEqual(set(answer['answers'][key]), fields)
        for key in ('category', 'quality'):
            self.assertAlmostEqual(sum(answer['answers'][key]['probabilities'].values()), 1.)
        score = answer['answers']['quality']
        self.assertAlmostEqual(score['score'], sum(int(k)*p for k,p in score['probabilities'].items()))
        self.assertEqual(score['legend']['1'], {'ask':'Concrete'})
        self.assertEqual(answer['answers']['category']['choice'], 'dev_tool')
    def test_invalid_logits_and_temperature_rejected(self):
        r = self.request(); z = {'inquiry': [0.,3.], 'category': [0.,4.,1.], 'quality': [0.,1.,2.]}
        for t in (0., -1., math.nan):
            with self.assertRaises(ValueError): response_from_logits(r,z,'test',t)
        z['quality'][0] = math.nan
        with self.assertRaises(ValueError): response_from_logits(r,z,'test')
    def test_unknown_fields_and_nonfinite_json_rejected(self):
        r = self.request(); r['questions']['category']['answer'] = 'dev_tool'
        with self.assertRaises(ValueError): validate_request(r)
        r = self.request(); r['state'] = {'x': math.inf}
        with self.assertRaises(ValueError): validate_request(r)
        with self.assertRaises(ValueError): validate_request({'model_input': {}})

if __name__ == '__main__': unittest.main()
