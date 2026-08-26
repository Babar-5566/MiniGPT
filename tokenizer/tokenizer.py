import json
class BPETokenizer:
    """Small corpus-trained byte-pair tokenizer; merges frequent adjacent symbols."""
    unk_token = '<UNK>'

    def __init__(self,vocab_size=512):
        self.vocab_size=vocab_size; self.tokens=[]; self.merges=[]; self.stoi={}; self.itos={}
        self.unk_id=None
    def train(self,text):
        seq=list(text); vocab=set(seq)
        while len(vocab)<self.vocab_size:
            pairs={}
            for a,b in zip(seq,seq[1:]): pairs[(a,b)]=pairs.get((a,b),0)+1
            if not pairs: break
            pair=max(pairs,key=pairs.get); new=''.join(pair); out=[]; i=0
            while i<len(seq):
                if i+1<len(seq) and (seq[i],seq[i+1])==pair: out.append(new); i+=2
                else: out.append(seq[i]); i+=1
            seq=out
            vocab.add(new); self.merges.append(pair)
            if len(vocab)>=self.vocab_size: break
        # Keep the special token outside the learned merge vocabulary.  It is
        # appended so token ids for existing vocabularies remain stable when
        # loading older tokenizer files.
        self.tokens=sorted(vocab)
        if self.unk_token not in self.tokens: self.tokens.append(self.unk_token)
        self.stoi={s:i for i,s in enumerate(self.tokens)}; self.itos={i:s for s,i in self.stoi.items()}; self.unk_id=self.stoi[self.unk_token]; return self
    def encode(self,text):
        seq=list(text)
        for pair in self.merges:
            out=[]; i=0
            while i<len(seq):
                if i+1<len(seq) and (seq[i],seq[i+1])==pair: out.append(''.join(pair)); i+=2
                else: out.append(seq[i]); i+=1
            seq=out
        # Unknown symbols map explicitly to <UNK>; do not silently substitute
        # a space (which can corrupt text and hide vocabulary mismatches).
        fallback=self.unk_id if self.unk_id is not None else self.stoi.get(self.unk_token)
        return [self.stoi.get(s,fallback) for s in seq]
    def decode(self,ids): return ''.join(self.itos.get(i,self.unk_token) for i in ids)
    def save(self,path): json.dump({'vocab_size':self.vocab_size,'tokens':self.tokens,'merges':self.merges},open(path,'w',encoding='utf8'))
    @classmethod
    def load(cls,path):
        d=json.load(open(path,encoding='utf8')); x=cls(d['vocab_size']); x.tokens=d['tokens']
        # Older tokenizer files did not contain <UNK>; append it to preserve
        # every pre-existing token id while enabling explicit fallback.
        if x.unk_token not in x.tokens: x.tokens.append(x.unk_token)
        x.merges=[tuple(p) for p in d['merges']]; x.stoi={s:i for i,s in enumerate(x.tokens)}; x.itos={i:s for s,i in x.stoi.items()}; x.unk_id=x.stoi[x.unk_token]; return x
