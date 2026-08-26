# Experiments

Presets live in `experiments/configs.json` and are selected with `--config tiny`, `--config small`, or `--config medium`. A run creates a unique directory:

```text
experiments/<config>-<timestamp>/
├── config.json
├── history.json
├── metrics.json
├── checkpoint_metadata.json
├── loss.png
└── checkpoints/
```

`history.json` records evaluation steps, training loss, validation loss, and perplexity. `metrics.json` records parameter count and training time. The checkpoint directory stores regular checkpoints and the best validation checkpoint. `loss.png` plots recorded train and validation loss. Compare runs by their `config.json`, `metrics.json`, and final history records; do not compare the 20-step smoke examples to converged results.

The verified 20-step examples are tiny (137,024 parameters, validation loss 36.3624), small (875,392, 27.6462), and medium (2,792,640, 23.9803). These are validation examples only.
