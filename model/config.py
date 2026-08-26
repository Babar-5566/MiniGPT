from dataclasses import dataclass, asdict

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    early_stopping_patience: int = 5
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        values = cls().to_dict()
        values.update(d)
        return cls(**values)
