## DETAILED

### MiniGPT — Quran-Corpus Decoder-Only Language Model

*Python, PyTorch, Streamlit, matplotlib, pytest*

- Built an end-to-end decoder-only Transformer language-model pipeline from validated Quran records through BPE-style tokenization, next-token training, checkpointing, and CLI/Streamlit inference.
- Implemented reproducible training with configurable seeds, Tiny/Small/Medium presets, warmup plus cosine learning-rate decay, validation-loss tracking, resumable checkpoints, and best-model selection.
- Developed controllable text generation with greedy decoding and temperature, top-k, and top-p sampling; verified behavior with automated sampling and tokenizer tests.
- Ran verified 2,000-step experiments on the same Quran dataset: Tiny (137,024 parameters, validation loss 5.8205), Small (875,392, 5.0820), and Medium (2,792,640, 4.4849), with per-run metrics, histories, checkpoints, and loss plots.


## USED

### MiniGPT — Decoder-Only Transformer Language Model from Scratch
**[GitHub](#)** | **[Live Demo](#)**
*Python, PyTorch, Streamlit, Matplotlib, pytest*

- Built an end-to-end decoder-only Transformer LM pipeline (BPE-style tokenization → training → checkpointing → CLI/Streamlit inference) trained from scratch on a structured text corpus.
- Engineered reproducible training with configurable seeds, warmup + cosine LR decay, validation tracking, and resumable checkpointing across three model-size presets (Tiny/Small/Medium).
- Implemented controllable generation (greedy, temperature, top-k, top-p sampling), validated with a pytest suite covering tokenizer correctness and sampling behavior.
- Ran controlled scaling experiments (2,000 steps/run): validation loss improved 23% (5.82 → 4.48) as parameters scaled 20× (137K → 2.79M), demonstrating the model-capacity/loss trade-off.