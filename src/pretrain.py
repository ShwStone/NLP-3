"""在 AG News 训练集上继续预训练作业二的解码器语言模型。

这一步只使用新闻正文做下一个词预测，不使用类别标签。得到的 checkpoint 作为
分类头和 Prompt 实验的共同初始化；它也是“加载已有语言模型权重”的实际演示。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .data import (ROOT, build_lm_stream, prepare_data, sample_lm_batch)
from .model import DecoderLM, ModelConfig, count_parameters, theoretical_parameters
from .utils import (Stopwatch, environment_info, precision_context,
                    seed_everything, write_json)


@torch.no_grad()
def evaluate(model, stream, device, batch_size, batches, seed):
    """固定种子重建评估窗口，保证每个评估点看到同一批窗口。"""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = sample_lm_batch(stream, batch_size, model.cfg.context_length,
                               generator, device)
        with precision_context(device):
            loss = model(x, y)[1]
        losses.append(loss.item())
    return sum(losses) / len(losses)


def learning_rate(step, total_steps, peak, warmup):
    if step <= warmup:
        return peak * step / max(1, warmup)
    fraction = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--max-doc-tokens", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--peak-lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(ROOT / "results" / "lm_pretrained.pt"))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.context_length, args.max_doc_tokens,
           args.eval_interval, args.eval_batches) < 1:
        parser.error("步数、batch、长度和评估参数必须为正")
    seed_everything(args.seed)
    device = torch.device(args.device)
    bundle = prepare_data()
    tokenizer = bundle.tokenizer
    train_stream = build_lm_stream(bundle.train, tokenizer, args.max_doc_tokens)
    validation_stream = build_lm_stream(bundle.validation, tokenizer, args.max_doc_tokens)
    cfg = ModelConfig(vocab_size=len(tokenizer), context_length=args.context_length,
                      n_layers=4, d_model=256, n_heads=4, dropout=0.1)
    model = DecoderLM(cfg).to(device)
    parameters = count_parameters(model)
    assert parameters["total"] == theoretical_parameters(cfg)
    # 语言模型输出头与词嵌入共享同一参数对象；parameters() 自动去重。
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8)
    settings = {
        "steps": args.steps, "batch_size": args.batch_size,
        "context_length": args.context_length, "max_document_tokens": args.max_doc_tokens,
        "peak_lr": args.peak_lr, "warmup_steps": args.warmup,
        "betas": [0.9, 0.95], "weight_decay_matrix": 0.1,
        "gradient_clip": 1.0, "seed": args.seed,
        "eval_interval": args.eval_interval, "eval_batches": args.eval_batches,
        "train_eval_seed": 142, "validation_eval_seed": 242,
        "precision": "BF16 autocast; FP32 parameters/optimizer" if device.type == "cuda" else "FP32",
    }
    environment = environment_info(device)
    print(json.dumps({"config": asdict(cfg), "parameters": parameters["total"],
                      "settings": settings, "environment": environment},
                     ensure_ascii=False, indent=2), flush=True)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    write_json(results / "lm_parameters.json", parameters)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = Stopwatch(device)
    evaluations = []
    best_validation = float("inf")
    best_state = None
    best_step = 0
    generator = torch.Generator().manual_seed(args.seed + 100)
    with (results / "lm_training_log.csv").open("w", newline="") as raw_file, \
         (results / "lm_eval_log.csv").open("w", newline="") as eval_file:
        writer = csv.DictWriter(raw_file, fieldnames=[
            "step", "loss", "learning_rate", "gradient_norm_before_clip", "elapsed_seconds"])
        writer.writeheader()
        eval_writer = csv.DictWriter(eval_file, fieldnames=[
            "step", "train_loss", "validation_loss", "train_perplexity",
            "validation_perplexity", "elapsed_seconds"])
        eval_writer.writeheader()

        def log_evaluation(step):
            nonlocal best_validation, best_state, best_step
            with started:
                train_loss = evaluate(model, train_stream, device, args.batch_size,
                                      args.eval_batches, 142)
                val_loss = evaluate(model, validation_stream, device, args.batch_size,
                                    args.eval_batches, 242)
            record = {
                "step": step, "train_loss": train_loss, "validation_loss": val_loss,
                "train_perplexity": math.exp(train_loss),
                "validation_perplexity": math.exp(val_loss),
                "elapsed_seconds": started.current(synchronize=True),
            }
            evaluations.append(record)
            eval_writer.writerow(record)
            eval_file.flush()
            print(f"step={step:4d} train={train_loss:.4f} val={val_loss:.4f} "
                  f"val_ppl={math.exp(val_loss):.2f} elapsed={started.current():.1f}s",
                  flush=True)
            if val_loss < best_validation:
                best_validation = val_loss
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}

        log_evaluation(0)
        model.train()
        with started:
            for step in range(1, args.steps + 1):
                lr = learning_rate(step, args.steps, args.peak_lr, args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, y = sample_lm_batch(train_stream, args.batch_size,
                                       args.context_length, generator, device)
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device):
                    loss = model(x, y)[1]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"第 {step} 步损失非有限值")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"],
                    error_if_nonfinite=True)
                optimizer.step()
                writer.writerow({
                    "step": step, "loss": loss.item(), "learning_rate": lr,
                    "gradient_norm_before_clip": norm.item(),
                    "elapsed_seconds": started.current(),
                })
                if step % args.eval_interval == 0 or step == args.steps:
                    raw_file.flush()
                    log_evaluation(step)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    checkpoint = {
        "config": asdict(cfg), "state_dict": model.state_dict(),
        "tokenizer": tokenizer.state_dict(), "step": args.steps,
        "best_step": best_step, "best_validation_loss": best_validation,
        "seed": args.seed, "data": {
            "train_csv_sha256": bundle.metadata["train_csv_sha256"],
            "test_csv_sha256": bundle.metadata["test_csv_sha256"],
            "counts": bundle.metadata["counts"],
        },
        "settings": settings,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    # 另外保存验证损失最优的权重，分类实验默认加载它。
    best_checkpoint = dict(checkpoint)
    best_checkpoint["state_dict"] = best_state
    best_checkpoint["step"] = best_step
    best_path = Path(args.output).with_name("lm_pretrained_best.pt")
    torch.save(best_checkpoint, best_path)
    summary = {
        "config": asdict(cfg), "parameters": parameters,
        "settings": settings, "environment": environment,
        "data": bundle.metadata, "initial": evaluations[0],
        "final": evaluations[-1], "best": evaluations[
            next(i for i, record in enumerate(evaluations)
                 if record["step"] == best_step)],
        "evaluations": evaluations,
        "training_wall_seconds_including_eval": started.elapsed,
        "peak_cuda_allocated_bytes": peak_memory,
        "processed_training_tokens": args.steps * args.batch_size * args.context_length,
        "train_stream_tokens": len(train_stream),
        "validation_stream_tokens": len(validation_stream),
        "checkpoint": str(args.output),
        "best_checkpoint": str(best_path),
        "loss_reduction_train_percent": 100 * (1 - evaluations[-1]["train_loss"] / evaluations[0]["train_loss"]),
        "loss_reduction_validation_percent": 100 * (1 - evaluations[-1]["validation_loss"] / evaluations[0]["validation_loss"]),
    }
    write_json(results / "lm_summary.json", summary)
    print("语言模型预训练完成。", flush=True)
    print(json.dumps({"initial": summary["initial"], "final": summary["final"],
                      "best": summary["best"], "seconds": started.elapsed},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
