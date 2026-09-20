"""Numerical checks for the reference scorer; requires requirements-training.txt."""
import unittest
import torch
from transformers import Qwen3Config,Qwen3Model

from scripts.model import BoundaryEmbedding,CandidateScorer


class ScorerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.base=Qwen3Model(Qwen3Config(vocab_size=600,hidden_size=32,intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            attention_dropout=0.0))
        for p in self.base.parameters():p.requires_grad_(False)
        self.embedding=BoundaryEmbedding(self.base.get_input_embeddings(),[590,591,592,593])
        self.base.set_input_embeddings(self.embedding)
        self.scorer=CandidateScorer(self.base,torch.nn.Linear(32,1,bias=False),0)

    def test_chunked_soft_distribution_loss_matches_direct_gradient(self):
        paths=[[590,2,9,592,5,593],[590,2,9,592,6,7,593],[590,2,9,592,8,593]]
        gradients=[]
        logits=[]
        for chunk,recompute in [(3,False),(1,True)]:
            self.scorer.zero_grad(set_to_none=True)
            z=self.scorer(paths,chunk,recompute)
            (-(torch.tensor([0.2, 0.6, 0.2]) * z.log_softmax(0)).sum()).backward()
            gradients.append([self.scorer.head.weight.grad.clone(),self.embedding.new_rows.grad.clone()])
            logits.append(z.detach())
        torch.testing.assert_close(logits[0],logits[1],atol=1e-6,rtol=1e-5)
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-4)

    def test_old_embedding_rows_not_changed_by_adamw(self):
        before=self.embedding.base.weight.detach().clone()
        optimizer=torch.optim.AdamW([p for p in self.scorer.parameters() if p.requires_grad],lr=.01,weight_decay=.1)
        self.scorer([[590,12,592,15,593],[590,12,592,19,593]],1).softmax(0)[0].backward()
        optimizer.step()
        self.assertTrue(torch.equal(before,self.embedding.base.weight))
        self.assertIsNone(self.embedding.base.weight.grad)



if __name__=='__main__':unittest.main()
