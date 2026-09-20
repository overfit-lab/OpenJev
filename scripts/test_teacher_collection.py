"""Soft-target mathematics and code mappings, without model weights or vLLM."""
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid
from scripts.chat_records import digest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from scripts.collect_teacher import average, distribution, task, validate_target, thinking_prompt, LocalTeacher, same_distribution
from scripts import collect_teacher
from scripts.chat_records import canonical
from scripts.question_data import make_record, write_json, sha, manifest


class CollectionTests(unittest.TestCase):
    def test_chat_template_is_encoded_explicitly_without_duplicate_special_tokens(self):
        tokenizer=Mock()
        tokenizer.apply_chat_template.return_value='<start>user<assistant><think>'
        tokenizer.encode.return_value=[1,2,3]
        self.assertEqual(thinking_prompt(tokenizer,{'state':'fixture'}),[1,2,3])
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['tokenize'])
        tokenizer.encode.assert_called_once_with('<start>user<assistant><think>',add_special_tokens=False)

    def test_incomplete_thinking_never_becomes_a_soft_target(self):
        teacher=LocalTeacher.__new__(LocalTeacher)
        teacher.tokenizer=Mock()
        teacher.tokenizer.encode.return_value=[1,2]
        teacher.args=SimpleNamespace(thinking_tokens=10,max_model_len=100)
        teacher.SamplingParams=lambda **kwargs:kwargs
        teacher.thinking_end_id=9
        teacher.model=Mock()
        teacher.model.generate.return_value=[SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7,8],finish_reason='length')])]
        with self.assertRaisesRegex(ValueError,'thinking_incomplete'):teacher.trace({}, {}, 42)
        self.assertEqual(teacher.model.generate.call_count,1)

    def test_logits_are_read_before_answer_and_use_all_candidate_ids(self):
        teacher=LocalTeacher.__new__(LocalTeacher)
        teacher.tokenizer=Mock()
        teacher.tokenizer.encode.side_effect=[[1,2],[6],[1,2,7,9,6],[1,2,7,9,6,0],[1,2,7,9,6,2]]
        teacher.tokenizer.decode.return_value='prefix'
        teacher.args=SimpleNamespace(thinking_tokens=10,max_model_len=100)
        teacher.SamplingParams=lambda **kwargs:kwargs
        teacher.thinking_end_id=9;teacher.vocab_size=3
        teacher.model=Mock()
        teacher.model.generate.side_effect=[
            [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7,9],finish_reason='stop')])],
            [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{i:SimpleNamespace(logprob=z) for i,z in enumerate([2.,-1.,0.])}])])]]
        result=teacher.trace({}, {'AA':{'name':'a','token_id':0},'AB':{'name':'b','token_id':2}}, 42)
        call=teacher.model.generate.call_args
        self.assertEqual(call.args[0],[{'prompt_token_ids':[1,2,7,9,6]}])
        self.assertEqual(call.args[1]['logprobs'],-1)
        self.assertEqual(call.args[1]['max_tokens'],1)
        self.assertEqual(result['candidate_logits'],{'a':2.,'b':0.})
        self.assertGreater(result['legal_candidate_mass'],.9)

    def test_collection_replay_resume_and_tampered_target_rejection(self):
        root=Path('outputs')/('test_collection_'+uuid.uuid4().hex)
        model=root/'model';questions=root/'questions';book=root/'book'
        for p in (model,questions,book):p.mkdir(parents=True)
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:write_json(model/name,{})
        write_json(book/'codebook.json',{'tokenizer_sha256':sha(model/'tokenizer.json'),
                   'codes':[{'code':'AA','token_id':1},{'code':'AB','token_id':2}]})
        manifest(book)
        prefix={'id':'p','group_id':'g','split':'train','source':'fixture','source_family':'fixture','original_prefix_sha256':'fixture'}
        request={'state':'fixture','questions':{'q':{'type':'noul','instructions':'Is evidence visible?'}}}
        row=make_record(prefix,request,{}, {}, {})
        (questions/'questions.jsonl').write_text(canonical(row)+'\n');manifest(questions)
        args=SimpleNamespace(model=model,questions=questions,codebook=book,output=root/'run',limit=1,
            traces=4,thinking_tokens=10,max_model_len=100,tensor_parallel=1,gpu_memory=.65,seed=42,resume=False)
        fake=Mock()
        fake.trace.side_effect=lambda inputs,mapping,seed:{'seed':seed,'answer_prefix_verified':True,
            'score_mode':'raw_logits','candidate_logits':{'false':-1.,'true':2.},'legal_candidate_mass':.9}
        try:
            with patch.object(collect_teacher,'LocalTeacher',return_value=fake),\
                 patch.object(collect_teacher.importlib.metadata,'version',return_value='fixture'),redirect_stdout(io.StringIO()):
                collect_teacher.run(args)
                args.resume=True;collect_teacher.run(args)
                self.assertEqual(fake.trace.call_count,4)
                self.assertEqual(collect_teacher.validate_collection(args.output,questions,book)['verified_soft_records'],1)
                p=next((args.output/'records').glob('*.json'));record=json.loads(p.read_text())
                record['target']['q']['probabilities']={'false':.5,'true':.5}
                write_json(p,record);manifest(args.output)
                with self.assertRaisesRegex(ValueError,'replay_mismatch'):
                    collect_teacher.validate_collection(args.output,questions,book)
        finally:
            archive=Path('trash')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_')
            archive.mkdir(exist_ok=True);shutil.move(str(root),str(archive/root.name))

    def test_stable_softmax_and_temperature_preserve_ranking(self):
        scores={'a':10000.,'b':9990.,'c':9980.}
        cold=distribution(scores,1);warm=distribution(scores,4)
        self.assertAlmostEqual(sum(cold.values()),1.)
        self.assertAlmostEqual(sum(warm.values()),1.)
        self.assertGreater(cold['a'],warm['a'])
        self.assertGreater(warm['a'],warm['b'])
        with self.assertRaises(ValueError): distribution({'a':float('nan')},1)
        with self.assertRaises(ValueError): distribution(scores,0)

    def test_average_maps_semantic_names_not_position(self):
        self.assertEqual(average([{'a':.9,'b':.1},{'b':.7,'a':.3}]),{'a':.6,'b':.39999999999999997})
        with self.assertRaises(ValueError): average([{'a':1.},{'b':1.}])
        self.assertTrue(same_distribution({'a':.6},{'a':.6000000000000001}))
        self.assertFalse(same_distribution({'a':.6},{'a':.61}))
        self.assertFalse(same_distribution({'a':float('nan')},{'a':.6}))

    def test_255_candidates_do_not_use_26_letter_limit(self):
        row={'input':{'state':'fixture','questions':{'q':{'type':'choice','instructions':'Pick',
            'criteria':{f'candidate_{i}':f'Definition {i}' for i in range(255)}}}}}
        book={'codes':[{'code':chr(65+i//26)+chr(65+i%26),'token_id':i} for i in range(255)]}
        _,mapping,inputs=task(row,book,42)
        self.assertEqual(len(mapping),255)
        self.assertEqual(len({v['token_id'] for v in mapping.values()}),255)
        self.assertEqual({v['name'] for v in mapping.values()},set(row['input']['questions']['q']['criteria']))
        self.assertNotIn('target',inputs)

    def test_soft_target_requires_exact_candidate_coverage_and_identity(self):
        q={'id':'q','input':{'state':'fixture','questions':{'intent':{'type':'noul','instructions':'Visible?'}}}}
        r={'question_record_id':'q','input_sha256':digest(q['input']),
           'target':{'intent':{'label_type':'teacher_soft','probabilities':{'false':.3,'true':.7}}},'training_eligible':False}
        validate_target(r,q)
        for mode in ['missing','wrong_input','approved','hard']:
            changed=deepcopy(r)
            if mode=='missing':del changed['target']['intent']['probabilities']['false']
            elif mode=='wrong_input':changed['input_sha256']='different'
            elif mode=='approved':changed['training_eligible']=True
            else:changed['target']['intent']['label_type']='teacher_hard'
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_target(changed,q)


if __name__=='__main__':unittest.main()
