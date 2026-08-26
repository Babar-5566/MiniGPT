# Development

Work from `C:\Tiny LLM` with Python 3.12. Install dependencies with `python -m pip install -r requirements.txt` and pytest with `python -m pip install pytest`.

Run checks with:

```powershell
pytest -q
python -m py_compile model/transformer.py training/train.py inference/generate.py
```

The current suite contains 5 tests. Keep changes scoped, preserve the Transformer architecture, and add a focused test for behavior changes. Training artifacts belong under timestamped `experiments/` directories. Do not commit large generated checkpoints unless intentionally publishing an example.
