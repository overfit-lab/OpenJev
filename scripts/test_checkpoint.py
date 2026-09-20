"""PiSSA reload and joint updates on two tiny local backbone architectures."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import unittest
import uuid
import torch
from transformers import Qwen3Config,Qwen3Model,LlamaConfig,LlamaModel,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers
from scripts.contract import BOUNDARIES,compile_request
from scripts.model import build,save_checkpoint,load_checkpoint,predict

class CheckpointTest(unittest.TestCase):
    def test_joint_pissa_updates_and_reload_on_two_backbones(self):
        root=Path('outputs')/('test_checkpoint_'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            for family in ('qwen','llama'):
                with self.subTest(family=family):
                    raw=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'one':2,'two':3,'state':4,'choice':5,'Pick':6},unk_token='[UNK]'))
                    raw.pre_tokenizer=pre_tokenizers.Whitespace()
                    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]')
                    tokenizer.add_special_tokens({'additional_special_tokens':BOUNDARIES})
                    common=dict(vocab_size=7,hidden_size=32,intermediate_size=48,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
                    base=Qwen3Model(Qwen3Config(**common)) if family=='qwen' else LlamaModel(LlamaConfig(**common))
                    identity={'model':'openjev-test/'+family,'revision':'fixture'}
                    config={'adapter':{'r':4,'lora_alpha':8,'lora_dropout':0.,'target_modules':'all-linear','bias':'none'}}
                    scorer,rows=build(base,tokenizer,'pissa',config,23,identity)
                    request={'state':'state','questions':{
                        'q':{'type':'choice','instructions':'Pick','criteria':{'one':'one','two':'two'}},
                        'n':{'type':'noul','instructions':'one?'},
                        's':{'type':'score','instructions':'Pick','criteria':['one','two']}}}
                    compiled=compile_request(tokenizer,request)
                    frozen=rows.base.weight.detach().clone()
                    before={n:p.detach().clone() for n,p in scorer.named_parameters() if p.requires_grad}
                    optimizer=torch.optim.AdamW([p for p in scorer.parameters() if p.requires_grad],lr=.01)
                    z=scorer(compiled[0].paths,1)
                    (-(torch.tensor([0.8, 0.2]) * z.log_softmax(0)).sum()).backward()
                    optimizer.step()
                    self.assertTrue(torch.equal(frozen,rows.base.weight))
                    changed=[n for n,p in scorer.named_parameters() if n in before and not torch.equal(before[n],p)]
                    self.assertTrue(any('lora_' in n for n in changed))
                    self.assertIn('head.weight',changed)
                    self.assertTrue(any('new_rows' in n for n in changed))
                    meta={**identity,'method':'pissa','output_model':'openjev-test'}
                    expected=predict(scorer,tokenizer,meta,request)
                    destination=root/family
                    save_checkpoint(scorer,rows,tokenizer,destination,meta)
                    restored,_,loaded,metadata=load_checkpoint(destination)
                    actual=predict(restored,loaded,metadata,request)
                    for q in compiled:
                        with torch.no_grad():
                            a=scorer(q.paths,1,False);b=restored(q.paths,1,False)
                        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
                    self.assertEqual(expected['answers'],actual['answers'])
                    self.assertEqual(set(actual),{'model','answers','usage'})
                    for p in destination.rglob('*.json'):
                        text=p.read_text()
                        self.assertNotIn(str(Path.cwd()),text)
                        self.assertNotIn(str(Path.home()),text)
        finally:
            archive=Path('trash')/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_')
            archive.mkdir(exist_ok=True)
            shutil.move(root,archive/root.name)

if __name__=='__main__':unittest.main()
