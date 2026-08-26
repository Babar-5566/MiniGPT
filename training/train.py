import argparse,json,math,random,time
from datetime import datetime
from pathlib import Path
import torch
from model import GPTConfig,MiniGPT
from tokenizer import BPETokenizer
from data.dataset import QuranDataset
from training.checkpoint import load_checkpoint,save_checkpoint

def load_named_config(name,path='experiments/configs.json'):
 if not name:return {}
 configs=json.load(open(path,encoding='utf8'))
 if name not in configs: raise ValueError(f'Unknown config {name!r}')
 return configs[name]
def create_experiment_dir(name,root='experiments'):
 root=Path(root); root.mkdir(parents=True,exist_ok=True); base=f'{name or "default"}-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'; run=root/base; suffix=1
 while run.exists(): run=root/f'{base}-{suffix}'; suffix+=1
 run.mkdir(); (run/'checkpoints').mkdir(); return run
def write_json(path,value): json.dump(value,open(path,'w',encoding='utf8'),indent=2)
def plot_history(history,path):
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 fig,ax=plt.subplots(); ax.plot([x['step'] for x in history],[x['train_loss'] for x in history],label='train'); ax.plot([x['step'] for x in history],[x['val_loss'] for x in history],label='validation'); ax.legend(); fig.savefig(path); plt.close(fig)
def tokenizer_from_state(s):
 t=BPETokenizer(s.get('vocab_size',512)); t.__dict__.update(s); t.merges=[tuple(p) for p in t.merges]; t.stoi={x:i for i,x in enumerate(t.tokens)}; t.itos={i:x for x,i in t.stoi.items()}; t.unk_id=t.stoi.get(t.unk_token); return t
def learning_rate(step,cfg):
 if step<cfg.warmup_iters:return cfg.learning_rate*(step+1)/cfg.warmup_iters
 return cfg.min_learning_rate
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='data/quran.txt'); p.add_argument('--steps',type=int); p.add_argument('--resume'); p.add_argument('--seed',type=int,default=1337); p.add_argument('--config',choices=['tiny','small','medium']); a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
 saved=load_checkpoint(a.resume) if a.resume else None; ds=QuranDataset(a.data); tok=tokenizer_from_state(saved['tokenizer']) if saved else BPETokenizer(512).train(ds.text()); ids=torch.tensor(tok.encode(ds.text())); n=int(.9*len(ids)); tr,va=ids[:n],ids[n:]
 if saved: cfg=GPTConfig.from_dict(saved['config'])
 else: cfg=GPTConfig(vocab_size=len(tok.tokens),max_iters=a.steps or 2000,**load_named_config(a.config))
 if len(tr)<=cfg.block_size or len(va)<=cfg.block_size: raise ValueError('Sequences must exceed block_size')
 run=create_experiment_dir(a.config or 'default'); write_json(run/'config.json',cfg.to_dict()); m=MiniGPT(cfg); opt=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate); start=0; hist=[]
 def batch(d):
  ix=torch.randint(0,len(d)-cfg.block_size,(cfg.batch_size,)); return torch.stack([d[i:i+cfg.block_size] for i in ix]),torch.stack([d[i+1:i+cfg.block_size+1] for i in ix])
 def val():
  m.eval(); out=[]
  with torch.no_grad():
   for _ in range(cfg.eval_iters): out.append(m(*batch(va))[1].item())
  return sum(out)/len(out)
 t=time.time()
 for step in range(start,a.steps or cfg.max_iters):
  m.train(); x,y=batch(tr); _,loss=m(x,y); opt.zero_grad(); loss.backward(); opt.step()
  if step%cfg.eval_interval==0 or step==(a.steps or cfg.max_iters)-1:
   v=val(); rec={'step':step,'train_loss':loss.item(),'val_loss':v}; hist.append(rec); print(rec); save_checkpoint(run/'checkpoints'/f'checkpoint_{step}.pt',m,opt,cfg,tok,step,hist); save_checkpoint(run/'checkpoints'/'best_model.pt',m,opt,cfg,tok,step,hist)
 write_json(run/'history.json',hist); write_json(run/'metrics.json',{'parameter_count':m.parameter_count(),'training_time_sec':time.time()-t}); write_json(run/'checkpoint_metadata.json',{'count':len(hist)}); plot_history(hist,run/'loss.png'); print('experiment',run)
if __name__=='__main__':main()
