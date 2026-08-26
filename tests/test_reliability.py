import json
import pytest
import torch
from data.dataset import QuranDataset, load_quran, parse_juz_mappings
from model import GPTConfig, MiniGPT
from tokenizer import BPETokenizer
from training.checkpoint import save_checkpoint, load_checkpoint
from training.train import load_named_config, create_experiment_dir

def test_tokenizer_roundtrip_and_unk(tmp_path):
    t=BPETokenizer(16).train('abc abc')
    assert t.decode(t.encode('abc abc'))=='abc abc'
    assert t.encode('!')[0]==t.unk_id
    p=tmp_path/'tok.json'; t.save(p); loaded=BPETokenizer.load(p)
    assert loaded.decode(loaded.encode('abc abc'))=='abc abc'

def test_dataset_validation(tmp_path):
    q=tmp_path/'q.txt'; q.write_text('1|1|a\n1|1|duplicate\n0|2|bad\n',encoding='utf8')
    with pytest.raises(ValueError,match='[Dd]uplicate'): QuranDataset(q, tmp_path/'j.txt')

def test_checkpoint_save_load(tmp_path):
    t=BPETokenizer(8).train('abcdef')
    c=GPTConfig(block_size=4,vocab_size=len(t.tokens),n_embd=8,n_head=2,n_layer=1)
    m=MiniGPT(c); o=torch.optim.AdamW(m.parameters()); p=tmp_path/'c.pt'; save_checkpoint(p,m,o,c,t,2,{'val_loss':1})
    assert load_checkpoint(p)['step']==2

def test_generation_sampling():
    logits=torch.tensor([[4.,3.,2.,1.]])
    assert torch.isfinite(MiniGPT.filter_logits(logits,top_k=2)).sum()==2
    assert not torch.equal(MiniGPT.filter_logits(logits,.5),MiniGPT.filter_logits(logits,2.))

def test_experiment_config_and_storage(tmp_path):
 p=tmp_path/'configs.json'; p.write_text('{"tiny":{"n_embd":64}}',encoding='utf8')
 assert load_named_config('tiny',p)['n_embd']==64
 a=create_experiment_dir('tiny',tmp_path); b=create_experiment_dir('tiny',tmp_path)
 assert a!=b and (a/'checkpoints').is_dir() and (b/'checkpoints').is_dir()
