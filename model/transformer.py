import math, torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd); self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout); self.register_buffer('mask', torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1,1,cfg.block_size,cfg.block_size))
    def forward(self, x):
        b,t,c=x.shape; q,k,v=self.qkv(x).split(c, dim=-1)
        q=q.view(b,t,self.n_head,self.head_dim).transpose(1,2); k=k.view(b,t,self.n_head,self.head_dim).transpose(1,2); v=v.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        a=(q @ k.transpose(-2,-1))/math.sqrt(self.head_dim); a=a.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf')); a=self.drop(F.softmax(a,dim=-1))
        y=(a@v).transpose(1,2).contiguous().view(b,t,c); return self.drop(self.proj(y))
class MLP(nn.Module):
    def __init__(self,cfg): super().__init__(); self.net=nn.Sequential(nn.Linear(cfg.n_embd,4*cfg.n_embd),nn.GELU(),nn.Linear(4*cfg.n_embd,cfg.n_embd),nn.Dropout(cfg.dropout))
    def forward(self,x): return self.net(x)
class Block(nn.Module):
    def __init__(self,cfg): super().__init__(); self.ln1=nn.LayerNorm(cfg.n_embd); self.attn=CausalSelfAttention(cfg); self.ln2=nn.LayerNorm(cfg.n_embd); self.mlp=MLP(cfg)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.mlp(self.ln2(x))
class MiniGPT(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.tok=nn.Embedding(cfg.vocab_size,cfg.n_embd); self.pos=nn.Embedding(cfg.block_size,cfg.n_embd); self.blocks=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]); self.ln=nn.LayerNorm(cfg.n_embd); self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False); self.head.weight=self.tok.weight
    def forward(self,idx,targets=None):
        _,t=idx.shape; assert t<=self.cfg.block_size; x=self.tok(idx)+self.pos(torch.arange(t,device=idx.device)); logits=self.head(self.ln(self.blocks(x))); loss=None if targets is None else F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)); return logits,loss
    @staticmethod
    def filter_logits(logits,temperature=1.,top_k=None,top_p=None):
        if temperature<=0: raise ValueError('temperature must be greater than zero')
        logits=logits/temperature
        if top_k is not None:
            if top_k<=0: raise ValueError('top_k must be greater than zero')
            cutoff=torch.topk(logits,min(top_k,logits.size(-1)),dim=-1).values[...,-1,None]
            logits=logits.masked_fill(logits<cutoff,float('-inf'))
        if top_p is not None:
            if not 0<top_p<=1: raise ValueError('top_p must be in (0, 1]')
            sorted_logits,sorted_indices=torch.sort(logits,descending=True,dim=-1)
            sorted_probs=torch.softmax(sorted_logits,dim=-1)
            remove=torch.cumsum(sorted_probs,dim=-1)>top_p
            remove[...,1:]=remove[...,:-1].clone(); remove[...,0]=False
            sorted_logits=sorted_logits.masked_fill(remove,float('-inf'))
            logits=torch.full_like(logits,float('-inf')).scatter(-1,sorted_indices,sorted_logits)
        return logits
    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.,top_k=None,top_p=None,sample=True):
        for _ in range(max_new_tokens):
            logits,_=self(idx[:,-self.cfg.block_size:]); logits=logits[:,-1,:]
            if sample:
                logits=self.filter_logits(logits,temperature,top_k,top_p)
                nxt=torch.multinomial(torch.softmax(logits,dim=-1),1)
            else: nxt=torch.argmax(logits,dim=-1,keepdim=True)
            idx=torch.cat((idx,nxt),1)
        return idx
    def parameter_count(self): return sum(p.numel() for p in self.parameters())
