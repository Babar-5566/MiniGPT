import os
import torch

def save_checkpoint(path,model,optimizer,cfg,tokenizer,step,metrics):
    if cfg.vocab_size != len(tokenizer.tokens):
        raise ValueError(f'Config vocab_size ({cfg.vocab_size}) does not match tokenizer vocabulary ({len(tokenizer.tokens)})')
    parent=os.path.dirname(os.path.abspath(path))
    os.makedirs(parent,exist_ok=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'config':cfg.to_dict(),'tokenizer':tokenizer.__dict__,'step':step,'metrics':metrics},path)

def load_checkpoint(path,model=None,optimizer=None,map_location='cpu'):
    if not path or not os.path.isfile(path): raise FileNotFoundError(f'Checkpoint not found: {path}')
    d=torch.load(path,map_location=map_location)
    required={'model','config','tokenizer','step','metrics'}
    missing=required-d.keys()
    if missing: raise ValueError(f'Invalid checkpoint; missing: {", ".join(sorted(missing))}')
    config_vocab=d['config'].get('vocab_size')
    tokenizer_vocab=len(d['tokenizer'].get('tokens', []))
    if config_vocab != tokenizer_vocab:
        raise ValueError(f'Checkpoint config vocab_size ({config_vocab}) does not match tokenizer vocabulary ({tokenizer_vocab})')
    if model is not None: model.load_state_dict(d['model'])
    if optimizer is not None and d.get('optimizer') is not None: optimizer.load_state_dict(d['optimizer'])
    return d
