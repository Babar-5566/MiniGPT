import argparse,torch,random
from model import GPTConfig,MiniGPT
from tokenizer import BPETokenizer
from training.checkpoint import load_checkpoint
def generate(prompt,checkpoint='checkpoints/checkpoint_1999.pt',max_new_tokens=200,temperature=0.8,top_k=None,top_p=None,sample=True,seed=None):
 if seed is not None: random.seed(seed); torch.manual_seed(seed)
 d=load_checkpoint(checkpoint); cfg=GPTConfig.from_dict(d['config']); tok=BPETokenizer(1); tok.__dict__.update(d['tokenizer']); tok.merges=[tuple(p) for p in tok.merges]; tok.stoi={s:i for i,s in enumerate(tok.tokens)}; tok.itos={i:s for s,i in tok.stoi.items()}; tok.unk_id=tok.stoi.get(tok.unk_token); m=MiniGPT(cfg); m.load_state_dict(d['model']); m.eval(); x=torch.tensor([tok.encode(prompt)]); return tok.decode(m.generate(x,max_new_tokens,temperature,top_k,top_p,sample)[0].tolist())
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('prompt'); p.add_argument('--checkpoint',default='checkpoints/checkpoint_1999.pt'); p.add_argument('--tokens',type=int,default=200); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int); p.add_argument('--top-p',type=float); p.add_argument('--greedy',action='store_true'); p.add_argument('--seed',type=int); a=p.parse_args(); print(generate(a.prompt,a.checkpoint,a.tokens,a.temperature,a.top_k,a.top_p,not a.greedy,a.seed))
