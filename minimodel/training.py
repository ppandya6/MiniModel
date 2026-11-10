"""Reproducible training, held-out evaluation, and portable checkpoints."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import random
import time
import torch
from .model import ModelConfig, TransformerLM
from .tokenizer import Tokenizer

@dataclass
class TrainConfig:
    data: str = "data/tiny.txt"
    output: str = "runs/baseline"
    tokenizer: str = "char"
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    expansion: int = 4
    tie_embeddings: bool = True
    batch_size: int = 16
    steps: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_steps: int = 10
    eval_interval: int = 50
    eval_batches: int = 10
    validation_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.eval_interval, self.eval_batches, self.threads) < 1:
            raise ValueError("steps, batch size, evaluation settings and threads must be positive")
        if not 0 < self.validation_fraction < 1 or self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_steps < 0:
            raise ValueError("invalid training configuration")


def batches(tokens, context, batch_size, generator, device):
    starts = torch.randint(len(tokens) - context, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + context] for i in starts])
    y = torch.stack([tokens[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens, batch_size=16, count=10, seed=123):
    """Mean NLL over fixed, sampled windows; tokens may overlap."""
    if len(tokens) <= model.cfg.context or min(batch_size, count) < 1:
        raise ValueError("evaluation requires positive settings and more tokens than context")
    was_training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    try:
        losses = [model(*batches(tokens, model.cfg.context, batch_size, rng, next(model.parameters()).device))[1].item() for _ in range(count)]
        nll = sum(losses) / len(losses)
        return {"loss": nll, "perplexity": math.exp(min(nll, 700)), "evaluated_tokens": batch_size * count * model.cfg.context}
    finally:
        model.train(was_training)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TransformerLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, Tokenizer(**checkpoint["tokenizer"]), checkpoint


def train(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    text = Path(cfg.data).read_text(encoding="utf-8")
    split = int(len(text) * (1 - cfg.validation_fraction))
    training_text, validation_text = text[:split], text[split:]
    tokenizer = Tokenizer.fit(training_text, cfg.tokenizer)
    train_ids, val_ids = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in (training_text, validation_text)]
    if min(len(train_ids), len(val_ids)) <= cfg.context:
        raise ValueError("both text splits must contain more tokens than context; add data or reduce context")
    mc = ModelConfig(tokenizer.vocab_size, cfg.context, cfg.width, cfg.heads, cfg.layers, cfg.dropout, cfg.expansion, cfg.tie_embeddings)
    model = TransformerLM(mc).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists() or (out / "best.pt").exists():
        raise ValueError("output already contains a run; choose a new output directory")
    metadata = {"train_config": asdict(cfg), "model_config": asdict(mc), "parameters": sum(p.numel() for p in model.parameters()), "data_sha256": hashlib.sha256(text.encode()).hexdigest(), "train_tokens": len(train_ids), "validation_tokens": len(val_ids), "validation_unknown_fraction": float((val_ids == 0).float().mean()) if cfg.tokenizer == "char" else 0, "torch_version": str(torch.__version__)}
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rng = torch.Generator().manual_seed(cfg.seed)
    best = float("inf")
    started = time.monotonic()
    history = []
    for step in range(cfg.steps + 1):
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            record = {"step": step, "train": evaluate(model, train_ids, cfg.batch_size, cfg.eval_batches), "validation": evaluate(model, val_ids, cfg.batch_size, cfg.eval_batches), "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"step {step:5d} train {record['train']['loss']:.4f} validation {record['validation']['loss']:.4f}", flush=True)
            checkpoint = {"model": model.state_dict(), "model_config": asdict(mc), "tokenizer": tokenizer.state(), "metadata": metadata, "step": step}
            torch.save(checkpoint, out / "last.pt")
            if record["validation"]["loss"] < best:
                best = record["validation"]["loss"]
                torch.save(checkpoint, out / "best.pt")
        if step == cfg.steps:
            break
        warmup = min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        progress = max(0, step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        lr = cfg.learning_rate * warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(*batches(train_ids, cfg.context, cfg.batch_size, rng, cfg.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return history
