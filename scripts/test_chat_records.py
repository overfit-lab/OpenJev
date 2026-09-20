"""Regression checks for current split isolation and conversation privacy."""
import itertools
import json
import random
import unittest
from scripts.freeze_chat_splits import allocate, threshold_join
from scripts.chat_records import sanitize_state, parse_json_response

class ChatRecordTests(unittest.TestCase):
    def test_threshold_join_matches_exhaustive_oracle(self):
        rng = random.Random(22)
        values = [set(rng.sample(range(30), rng.randint(1,30))) for _ in range(75)]
        for threshold in [.5,.8,.85,1.0]:
            actual = {tuple(sorted((a,b))) for a,b,_ in threshold_join(values,threshold)}
            expected = {(a,b) for a,b in itertools.combinations(range(len(values)),2)
                        if len(values[a]&values[b])/len(values[a]|values[b])>=threshold}
            self.assertEqual(actual,expected)


    def test_entire_holdout_family_including_mixed_group_is_protected(self):
        components = {str(i):{'families':{'main'}} for i in range(100)}
        components['0']['families']={'main','holdout'}
        components['1']['families']={'holdout'}
        config={'seed':4,'fractions':{'train':.8,'test':.2},'holdout_family':'holdout','holdout_split':'test'}
        assigned,counts=allocate(components,config)
        self.assertEqual(assigned['0'],'test')
        self.assertEqual(assigned['1'],'test')
        self.assertEqual(sum(s=='train' for s in assigned.values()),counts['train'])
        self.assertEqual(assigned,allocate(components,config)[0])


    def test_masking_preserves_identity_without_exporting_a_mapping(self):
        state={'messages':[{'role':'user','content':'a@example.org b@example.org a@example.org'}], 'hidden':'secret'}
        masked,counts=sanitize_state(state)
        content=masked['messages'][0]['content']
        self.assertEqual(content.count('[REDACTED_EMAIL_1]'),2)
        self.assertIn('[REDACTED_EMAIL_2]',content)
        self.assertNotIn('@',content)
        self.assertNotIn('hidden',masked)
        self.assertEqual(counts,{'EMAIL':2})


    def response(self,value):
        return {'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],'stop_reason':'end_turn'}


    def test_truncated_transport_is_not_a_valid_label(self):
        response=self.response({'decision':'accept'})
        response['stop_reason']='max_tokens'
        with self.assertRaises(ValueError): parse_json_response(response)



if __name__=="__main__":unittest.main()
