# MiniModel — a language model from scratch

A small PyTorch research lab for studying how decoder architecture, tokenization, data, and model size affect next-token prediction. Attention is implemented explicitly: no pretrained models, transformer libraries, or built-in transformer blocks.

## Quick start

Python 3.10+ is required. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU build; use the appropriate official PyTorch build for GPU work.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python -m unittest discover -s tests -v
minimodel train --config configs/tiny.json
minimodel generate runs/tiny/best.pt --prompt 'The ' --tokens 200
```

In the prepared cloud machine, activate `/workspace/.venvs/minimodel/bin/activate` instead. Training writes a configuration and dataset hash, JSONL loss history, and best/last model checkpoints to `runs/tiny`. Existing runs are protected against accidental overwrite; use `--output runs/new-name` to repeat an experiment. Checkpoints support inference, not exact optimizer-state resumption.

## Architecture

UTF-8 bytes or learned character IDs → token + learned position embeddings → repeated pre-normalized decoder blocks → final layer norm → vocabulary projection.

Each block applies multi-head scaled dot-product attention with a triangular causal mask, a residual connection, and a GELU feed-forward network with another residual connection. Dropout is applied to embeddings, attention weights, and residual branches. Input/output embeddings are tied by default. The objective is next-token cross entropy; optimization uses AdamW, gradient clipping, linear warmup, and cosine learning-rate decay.

- `minimodel/model.py`: embeddings, attention, decoder, sampling.
- `minimodel/tokenizer.py`: serializable character and byte tokenizers.
- `minimodel/training.py`: splitting, random windows, evaluation, training, checkpoints.
- `minimodel/cli.py`: train, generate, evaluate, and compare commands.
- `configs/`: tiny baseline, wider model, byte tokenizer.
- `tests/`: causality, gradients, overfitting, serialization, evaluation, and invalid configurations.

## Experiments

```bash
minimodel train --config configs/wide.json
minimodel train --config configs/byte.json
minimodel compare runs/tiny runs/wide runs/byte
# Supply a separate, untouched test file:
minimodel evaluate runs/tiny/best.pt --data path/to/test.txt --batches 50
```

Copy a JSON configuration to change `data`, `context`, `width`, `heads`, `layers`, `expansion`, `dropout`, `tie_embeddings`, `tokenizer`, `steps`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_steps`, `validation_fraction`, `seed`, `device`, or `threads`. All available fields and defaults are defined in `TrainConfig`. Width must be divisible by head count. Set `device` to `cuda` only with a compatible PyTorch build and available GPU. Defaults use CPU and two threads.

For architecture comparisons, hold the corpus, split, tokenizer, training budget, and seed constant. Compare parameter count, held-out loss, and elapsed time in `metrics.jsonl`. Repeat across several seeds before drawing conclusions. Equal steps do not mean equal compute, and a larger context changes the number of tokens processed per step.

For dataset experiments, provide separate UTF-8 corpora with controlled mixtures and matching budgets. By default the final 15% of raw characters is held out **before** fitting the tokenizer. For substantial work, construct document-level train/validation/test partitions externally to avoid related documents leaking across splits. The evaluation command uses every supplied file as held-out input; it does not remove the training portion for you.

Character mode learns its vocabulary only from training text and maps unseen characters to ID 0 (decoded as `�`). Byte mode covers all UTF-8 inputs without vocabulary fitting, although sampled byte sequences can be invalid UTF-8 and decode with replacement characters. Metadata reports the validation unknown-character fraction.

## Reading the results

Loss is mean negative log likelihood in nats per token; perplexity is `exp(loss)`. Evaluation uses a fixed seed and sampled overlapping windows for stable comparisons, not exhaustive corpus scoring or confidence intervals. Training evaluation disables dropout. Validation chooses the best checkpoint; a separate test set is needed for final reporting.

Per-token losses and perplexities **cannot be directly compared across different tokenizers**. Use a common tokenizer for quantitative comparisons or implement a common byte-normalized, exhaustive evaluation protocol. The bundled prose is an original smoke-test corpus, far too small to establish general language ability. Generated samples demonstrate mechanics, not benchmark performance. CPU seeding is repeatable in a fixed software environment; exact reproducibility across hardware or PyTorch releases is not guaranteed.

## Validation and scope

The test suite checks future-token isolation, finite gradients, learning a repeated sequence, tokenizer round trips, generation beyond the context window, checkpoint loading, deterministic evaluation, and output protection. CI also runs the complete tiny training and sampling workflow.

This is a compact research baseline: no distributed training, pretrained weights, web service, BPE, or optimizer-state resume. These can be added as separate experiments without obscuring the core implementation.

## License

MIT; see [LICENSE](LICENSE). The included synthetic corpus is covered by the same license.
