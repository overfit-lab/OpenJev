"""Question synthesis contracts, quotas and evidence validation; no API calls."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime
import io
import shutil
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from scripts.chat_records import parse_json_text, parse_json_response
from scripts.question_data import CHECKS, make_record, recipes, validate_draft, validate_record, validate_review
from scripts.test_contract import Tokenizer
from scripts import synthesize
from scripts.question_data import write_json


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/data/questions.json').read_text())
        self.state = {'messages':[{'role':'user','content':'Translate this document into French.'}]}
        self.recipe = {'type':'choice','k':2,'slot':0,'bucket':'2-8','language':'en','form':'direct','option_style':'short'}
        self.draft = {'questions':{'intent':{'type':'choice','instructions':'What action is requested?',
                                          'criteria':{'translate':'Translation','summarize':'Summary'}}}}

    def test_bare_and_single_fenced_json_without_permissive_repair(self):
        for text in ['{"questions":{}}','```json\n{"questions":{}}\n```','```\n{"questions":{}}\n```']:
            self.assertEqual(parse_json_text(text),{'questions':{}})
        invalid = ['{"a":1,"a":2}', '{"a":{"b":0,"b":1}}', '{"x":NaN}', '{"x":1e999}',
                   '[]','{} {}','Here is the JSON: {}','```python\n{}\n```','```json\n{}',
                   '```json\n{}\n```\n```json\n{}\n```']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError): parse_json_text(text)

    def test_recipe_has_exact_quotas_and_early_large_k(self):
        plan = recipes(self.config)
        self.assertEqual(Counter(r['bucket'] for r in plan),{'2-8':36,'9-32':20,'33-64':12,'65-128':8,'129-255':4,'noul':20})
        self.assertGreaterEqual(sum(r['k']==255 for r in plan),2)
        self.assertTrue(any(r['k']==255 for r in plan[:6]))
        self.assertEqual(plan,recipes(deepcopy(self.config)))

    def test_no_generator_labels_or_state_replacement(self):
        request,_ = validate_draft(self.draft,self.state,self.recipe,Tokenizer(),self.config)
        self.assertEqual(request['state'],self.state)
        for field in ['state','target','answer','probabilities','confidence']:
            draft = deepcopy(self.draft); draft[field] = 'injected'
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_draft(draft,self.state,self.recipe,Tokenizer(),self.config)

    def test_exact_k_and_duplicate_semantics(self):
        for criteria in [{'a':'Same','b':'ＳＡＭＥ'},{'a':'x',' A ':'y'},{'a':'x'}, {'a':'x','b':None}]:
            draft=deepcopy(self.draft);draft['questions']['intent']['criteria']=criteria
            with self.subTest(criteria=criteria),self.assertRaises(ValueError):
                validate_draft(draft,self.state,self.recipe,Tokenizer(),self.config)
        draft=deepcopy(self.draft);draft['questions']['intent']['criteria']={'c++':'C++','c#':'C#'}
        validate_draft(draft,self.state,self.recipe,Tokenizer(),self.config)

    def test_255_options_are_validated_as_a_complete_question(self):
        draft=deepcopy(self.draft);draft['questions']['intent']['criteria']={f'option_{i}':f'Description {i}' for i in range(255)}
        recipe={**self.recipe,'k':255};config={**self.config,'max_path_tokens':50000,'max_total_tokens':10000000}
        _,metrics=validate_draft(draft,self.state,recipe,Tokenizer(),config)
        self.assertEqual(len(metrics['option_tokens']),255)
        with self.assertRaises(ValueError):validate_draft(draft,self.state,recipe,Tokenizer(),{**config,'max_total_tokens':10})

    def test_review_requires_grounding_and_consistent_checks(self):
        request,_=validate_draft(self.draft,self.state,self.recipe,Tokenizer(),self.config)
        review={'decision':'accept','checks':dict.fromkeys(CHECKS,True),'issues':[],'evidence':['Translate this document']}
        self.assertTrue(validate_review(review,request))
        for variant in ['invented_quote','failed_check','hidden_label']:
            changed=deepcopy(review)
            if variant=='invented_quote':changed['evidence']=['not in conversation']
            elif variant=='failed_check':changed['checks']['grounded']=False
            else:changed['answer']='translate'
            with self.subTest(variant=variant),self.assertRaises(ValueError):validate_review(changed,request)

    def test_question_identity_and_protected_split_are_enforced(self):
        request,metrics=validate_draft(self.draft,self.state,self.recipe,Tokenizer(),self.config)
        prefix={'id':'p','group_id':'g','split':'train','source':'example.jsonl','source_family':'example','original_prefix_sha256':'fixture'}
        row=make_record(prefix,request,self.recipe,metrics,{'generation_request':'a','review_request':'b'})
        validate_record(row)
        self.assertNotIn('target',row)
        self.assertFalse(row['training_eligible'])
        for variant in ['input','split','approval']:
            changed=deepcopy(row)
            if variant=='input':changed['input']['state']['messages'][0]['content']='different'
            elif variant=='split':changed['provenance']['split']='test'
            else:changed['training_eligible']=True
            with self.subTest(variant=variant),self.assertRaises(ValueError):validate_record(changed)

    def test_resume_replays_accepted_jobs_without_relabeling_and_rejects_config_drift(self):
        root=Path('outputs')/('test_questions_'+uuid.uuid4().hex);root.mkdir(parents=True)
        config={**self.config,'question_count':2,'max_sources_per_slot':1,'workers':1,'forms':['direct']}
        config_file=root/'config.json';write_json(config_file,config)
        splits=root/'splits';splits.mkdir();write_json(splits/'manifest.json',{'files':{}})
        prefix={'id':'p','group_id':'g','split':'train','source':'fixture.jsonl','source_family':'fixture',
                'original_prefix_sha256':'fixture','state':self.state}
        prefixes=[{**prefix,'id':str(i),'group_id':str(i)} for i in range(4)]
        tokenizer=Tokenizer();tokenizer.fingerprint='fixture-tokenizer'
        calls=[]
        class Client:
            def __init__(self,output,config):
                self.output=output;self.stop=threading.Event();self.endpoint_fingerprint='fixture-endpoint'
            def scrub(self,value):return value
            def call(self,job_id,role,inputs,max_tokens):
                calls.append((job_id,role));rid=job_id+'_'+role
                if role=='generate':
                    payload={'questions':{'q':{'type':'choice','instructions':'Pick the relevant action.',
                        'criteria':{f'action_{i}':f'Distinct description {i}' for i in range(inputs['recipe']['k'])}}}}
                else:payload={'decision':'accept','checks':dict.fromkeys(CHECKS,True),'issues':[],'evidence':['Translate this document']}
                folder=self.output/'requests'/rid;folder.mkdir(parents=True,exist_ok=True)
                write_json(folder/'attempt-00.json',{'status':'ok','response':{'stop_reason':'end_turn',
                           'content':[{'type':'text','text':json.dumps(payload)}]}})
                return payload,rid
        args=SimpleNamespace(config=config_file,tokenizer='fixture',splits=splits,output=root/'run',max_jobs=1,resume=False)
        try:
            with patch.object(synthesize,'Tokenizer',return_value=tokenizer),patch.object(synthesize,'Client',Client),\
                 patch.object(synthesize,'prepare_prefixes',return_value=prefixes),redirect_stdout(io.StringIO()):
                synthesize.run(args)
                args.resume=True;synthesize.run(args)
                self.assertEqual(len(calls),4)
                synthesize.run(args);self.assertEqual(len(calls),4)
                self.assertEqual(len(synthesize.validate_run(args.output,'fixture')),2)
                write_json(config_file,{**config,'seed':0})
                with self.assertRaisesRegex(ValueError,'fingerprint'):synthesize.run(args)
        finally:
            archive=Path('trash')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_')
            archive.mkdir(exist_ok=True);shutil.move(str(root),str(archive/root.name))


if __name__=='__main__':unittest.main()
