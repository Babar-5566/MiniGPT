# Training

Training is launched from the project root with `python -m training.train`. The default split is 90% training and 10% validation. Each batch contains `block_size` inputs and the same sequence shifted by one token as targets. Cross-entropy is converted to perplexity with `exp(loss)`.

The learning rate warms up linearly for `warmup_iters`, then decays toward `min_learning_rate` using the configured schedule. Validation runs over `eval_iters` batches. Regular checkpoints and `best_model.pt` are saved when validation improves. `--resume` restores the saved model, optimizer, configuration, tokenizer, step, and metrics.

Examples:

```powershell
python -m training.train --config tiny --steps 2000 --seed 1337
python -m training.train --resume <checkpoint> --steps 500 --seed 1337
```

The model architecture is intentionally unchanged. CPU execution is supported but can be slow.
