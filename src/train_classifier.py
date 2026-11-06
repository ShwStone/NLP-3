"""在 AG News 上训练/微调解码器文本分类器。

三种模式：
  * ``full``   加载语言模型预训练权重，全参数微调；
  * ``frozen`` 加载预训练权重，冻结主干，只训练线性分类头（线性探针）；
  * ``scratch`` 不加载预训练权重，从随机初始化开始全参数训练（消融对照）。

分类头使用最后一个真实词元的隐状态；右侧 pad 不会被因果注意力看到，
因此变长 batch 的池化结果与逐条前向一致。验证集用于选择 checkpoint，
测试集只在训练结束后评估一次。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .classifier import (ClassifierModel, load_lm_checkpoint, parameter_report,
                         save_classifier_checkpoint)
from .data import (ROOT, EncodedTextDataset, LABELS, collate_classification,
                   prepare_data)
from .model import DecoderLM, ModelConfig
from .utils import (Stopwatch, classification_metrics, environment_info,
                    evaluate_classifier, precision_context, seed_everything,
                    write_json)


def make_loader(dataset, tokenizer, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda batch: collate_classification(batch, tokenizer.pad_id),
        generator=generator, num_workers=0, drop_last=False)


def cosine_with_warmup(step, total_steps, base_lr, warmup_steps, min_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * fraction)))


def build_optimizer(model: ClassifierModel, args):
    """矩阵参数衰减，偏置/LayerNorm 向量不衰减；主干与分类头使用不同学习率。"""
    groups = []
    backbone_decay, backbone_no_decay = [], []
    for name, parameter in model.backbone.named_parameters():
        (backbone_decay if parameter.ndim >= 2 else backbone_no_decay).append(parameter)
    if not model.backbone_frozen:
        groups.append({"params": backbone_decay, "weight_decay": args.weight_decay,
                       "lr": args.lr_backbone, "name": "backbone_decay"})
        groups.append({"params": backbone_no_decay, "weight_decay": 0.0,
                       "lr": args.lr_backbone, "name": "backbone_no_decay"})
    head_weight = [model.classification_head.weight]
    head_bias = [model.classification_head.bias]
    groups.append({"params": head_weight, "weight_decay": args.weight_decay,
                   "lr": args.lr_head, "name": "head_weight"})
    groups.append({"params": head_bias, "weight_decay": 0.0,
                   "lr": args.lr_head, "name": "head_bias"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "frozen", "scratch"], default="full")
    parser.add_argument("--lm-checkpoint", default=str(ROOT / "results" / "lm_pretrained_best.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None,
                        help="结果目录；默认 results/classifier_<mode>")
    parser.add_argument("--limit-train", type=int, default=0, help="仅用于快速调试")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        parser.error("epochs、batch-size 和 max-length 必须为正")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup-fraction 必须在 [0, 1) 内")
    output = Path(args.output) if args.output else ROOT / "results" / f"classifier_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)

    bundle = prepare_data(max_length=args.max_length)
    checkpoint, tokenizer, pretrained_backbone = load_lm_checkpoint(
        args.lm_checkpoint, map_location="cpu")
    lm_config = ModelConfig(**checkpoint["config"])
    if args.max_length > lm_config.context_length:
        raise ValueError("分类最大长度超过语言模型上下文长度")
    if args.mode != "scratch":
        # load_lm_checkpoint 已经严格加载了预训练权重；这里保留显式说明。
        print(f"已加载语言模型权重：{args.lm_checkpoint}（step={checkpoint.get('step')}）", flush=True)
    else:
        print("scratch 模式：保留相同词表和结构，但主干随机初始化。", flush=True)

    # scratch 对照必须真正丢弃预训练权重；只复用词表、模型结构和数据划分。
    backbone = DecoderLM(lm_config) if args.mode == "scratch" else pretrained_backbone
    model = ClassifierModel(backbone, len(LABELS), args.pooling, args.head_dropout)
    if args.mode == "frozen":
        model.set_backbone_frozen(True)
    model = model.to(device)
    params = parameter_report(model)

    train_dataset = EncodedTextDataset(bundle.train, tokenizer, args.max_length)
    validation_dataset = EncodedTextDataset(bundle.validation, tokenizer, args.max_length)
    test_dataset = EncodedTextDataset(bundle.test, tokenizer, args.max_length)
    if args.limit_train:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.limit_train, len(train_dataset))))
    train_loader = make_loader(train_dataset, tokenizer, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation_dataset, tokenizer, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, tokenizer, args.batch_size, False, args.seed)

    optimizer = build_optimizer(model, args)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    settings = {
        "mode": args.mode, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone if args.mode != "frozen" else 0.0,
        "lr_head": args.lr_head, "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction, "gradient_clip": args.gradient_clip,
        "max_length": args.max_length, "pooling": args.pooling,
        "head_dropout": args.head_dropout, "seed": args.seed,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "checkpoint_selection": "best validation accuracy; ties -> lower validation loss",
        "precision": "BF16 autocast; FP32 parameters/optimizer" if device.type == "cuda" else "FP32",
        "lm_checkpoint": str(args.lm_checkpoint),
        "lm_checkpoint_step": checkpoint.get("step") if args.mode != "scratch" else None,
        "train_examples": len(train_dataset), "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
    }
    environment = environment_info(device)
    print(json.dumps({"parameters": params, "settings": settings,
                      "environment": environment}, ensure_ascii=False, indent=2), flush=True)
    write_json(output / "settings.json", {"parameters": params, "settings": settings,
                                          "environment": environment,
                                          "model_config": asdict(lm_config),
                                          "data": bundle.metadata})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    history = []
    best = {"accuracy": -1.0, "loss": float("inf"), "epoch": 0, "state": None,
            "metrics": None}
    wall = Stopwatch(device)
    step = 0
    with (output / "training_log.csv").open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=[
            "epoch", "step", "train_loss", "train_accuracy",
            "validation_loss", "validation_accuracy", "validation_macro_f1",
            "learning_rate_backbone", "learning_rate_head", "elapsed_seconds"])
        writer.writeheader()
        with wall:
            for epoch in range(1, args.epochs + 1):
                model.train()
                running_loss, running_correct, seen = 0.0, 0, 0
                for input_ids, lengths, labels in train_loader:
                    input_ids = input_ids.to(device)
                    lengths = lengths.to(device)
                    labels = labels.to(device)
                    lr_backbone = cosine_with_warmup(step, total_steps, args.lr_backbone,
                                                     warmup_steps)
                    lr_head = cosine_with_warmup(step, total_steps, args.lr_head,
                                                 warmup_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_backbone if group["name"].startswith("backbone") else lr_head
                    optimizer.zero_grad(set_to_none=True)
                    with precision_context(device):
                        logits, loss = model(input_ids, lengths, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"第 {step} 步分类损失非有限值")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.gradient_clip, error_if_nonfinite=True)
                    optimizer.step()
                    batch = labels.numel()
                    running_loss += loss.item() * batch
                    running_correct += (logits.argmax(-1) == labels).sum().item()
                    seen += batch
                    step += 1
                    if step % 50 == 0 or step == total_steps:
                        print(f"epoch={epoch} step={step}/{total_steps} "
                              f"loss={running_loss / seen:.4f} "
                              f"acc={running_correct / seen:.4f} "
                              f"elapsed={wall.current():.1f}s", flush=True)
                validation = evaluate_classifier(model, validation_loader, device)
                validation_metrics = classification_metrics(
                    validation["targets"], validation["predictions"], LABELS)
                epoch_record = {
                    "epoch": epoch, "step": step,
                    "train_loss": running_loss / seen,
                    "train_accuracy": running_correct / seen,
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "elapsed_seconds": wall.current(synchronize=True),
                }
                history.append(epoch_record)
                writer.writerow({**epoch_record,
                                 "learning_rate_backbone": lr_backbone,
                                 "learning_rate_head": lr_head})
                log_file.flush()
                print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
                # 先比准确率，再比验证损失；保存 CPU 副本，避免后续训练覆盖。
                better = (validation_metrics["accuracy"] > best["accuracy"] or
                          (validation_metrics["accuracy"] == best["accuracy"] and
                           validation["loss"] < best["loss"]))
                if better:
                    best = {
                        "accuracy": validation_metrics["accuracy"],
                        "loss": validation["loss"], "epoch": epoch,
                        "metrics": validation_metrics,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()},
                    }
                    save_classifier_checkpoint(output / "best.pt", model, tokenizer, {
                        "mode": args.mode, "epoch": epoch, "step": step,
                        "validation_metrics": validation_metrics,
                        "settings": settings,
                    })
                    print(f"保存新的最佳 checkpoint（epoch={epoch}, "
                          f"val_acc={validation_metrics['accuracy']:.4f}）", flush=True)

    # 恢复验证集最优权重，再在测试集评估一次。
    model.load_state_dict(best["state"])
    model = model.to(device)
    if args.mode == "frozen":
        model.set_backbone_frozen(True)
    inference = Stopwatch(device)
    with inference:
        test_result = evaluate_classifier(model, test_loader, device)
    test_metrics = classification_metrics(test_result["targets"],
                                          test_result["predictions"], LABELS)
    test_metrics["loss"] = test_result["loss"]
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    metrics = {
        "mode": args.mode,
        "model_config": asdict(lm_config),
        "parameters": params,
        "settings": settings,
        "environment": environment,
        "data": {
            "counts": bundle.metadata["counts"],
            "train_csv_sha256": bundle.metadata["train_csv_sha256"],
            "test_csv_sha256": bundle.metadata["test_csv_sha256"],
            "max_length": args.max_length,
        },
        "history": history,
        "best_epoch": best["epoch"],
        "best_validation": best["metrics"],
        "best_validation_loss": best["loss"],
        "test": test_metrics,
        "training_wall_seconds": wall.elapsed,
        "inference_wall_seconds": inference.elapsed,
        "test_examples_per_second": len(test_dataset) / inference.elapsed,
        "peak_cuda_allocated_bytes": peak_memory,
        "checkpoint": str(output / "best.pt"),
        "checkpoint_sha256": hashlib.sha256(
            (output / "best.pt").read_bytes()).hexdigest(),
    }
    write_json(output / "metrics.json", metrics)
    # 保留测试集顺序的预测，便于生成样例和混淆矩阵。
    write_json(output / "test_predictions.json", {
        "targets": test_result["targets"],
        "predictions": test_result["predictions"],
    })
    print(json.dumps({"mode": args.mode, "parameters": params,
                      "best_epoch": best["epoch"],
                      "best_validation": {"accuracy": best["metrics"]["accuracy"]},
                      "test": {"accuracy": test_metrics["accuracy"],
                               "macro_f1": test_metrics["macro_f1"],
                               "loss": test_metrics["loss"]},
                      "training_seconds": wall.elapsed,
                      "inference_seconds": inference.elapsed},
                     ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
