import argparse
from dataclasses import fields
import json
from pathlib import Path
import torch
from .training import TrainConfig, train, load_checkpoint, evaluate


def main():
    parser = argparse.ArgumentParser(description="MiniModel transformer research lab")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train from a JSON configuration")
    training.add_argument("--config", required=True)
    training.add_argument("--output")
    sampling = commands.add_parser("generate")
    sampling.add_argument("checkpoint")
    sampling.add_argument("--prompt", default="The ")
    sampling.add_argument("--tokens", type=int, default=200)
    sampling.add_argument("--temperature", type=float, default=0.8)
    sampling.add_argument("--top-k", type=int, default=20)
    sampling.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", help="evaluate an entire supplied held-out file via sampled windows")
    evaluation.add_argument("checkpoint")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--batches", type=int, default=20)
    comparison = commands.add_parser("compare", help="summarize existing runs as JSON")
    comparison.add_argument("runs", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "train":
        values = json.loads(Path(args.config).read_text())
        unknown = set(values) - {f.name for f in fields(TrainConfig)}
        if unknown:
            parser.error(f"unknown configuration fields: {sorted(unknown)}")
        if args.output:
            values["output"] = args.output
        train(TrainConfig(**values))
    elif args.command == "compare":
        results = []
        for run in args.runs:
            path = Path(run)
            config = json.loads((path / "config.json").read_text())
            history = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
            best = min(history, key=lambda r: r["validation"]["loss"])
            results.append({"run": str(path), "parameters": config["parameters"], "tokenizer": config["train_config"]["tokenizer"], "data_sha256": config["data_sha256"], "best_step": best["step"], **best["validation"]})
        print(json.dumps(results, indent=2))
    else:
        model, tokenizer, _ = load_checkpoint(args.checkpoint)
        if args.command == "generate":
            if not args.prompt:
                parser.error("prompt must not be empty")
            ids = torch.tensor([tokenizer.encode(args.prompt)])
            result = model.generate(ids, args.tokens, args.temperature, args.top_k, torch.Generator().manual_seed(args.seed))
            print(tokenizer.decode(result[0].tolist()))
        else:
            text = Path(args.data).read_text(encoding="utf-8")
            ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            print(json.dumps(evaluate(model, ids, count=args.batches), indent=2))
