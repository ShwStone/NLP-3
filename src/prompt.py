"""Prompt 范式分类：不训练分类头，用语言模型给类别 verbalizer 打分。

模板（零样本）::

    title: <title> description: <description> topic: <verbalizer><eos>

四样本（每类一个训练集示例）时，在测试样本前拼接::

    topic: sports title: ... description: ...
    topic: business title: ... description: ...
    ...

对每个类别计算 ``log P(verbalizer, <eos> | prompt)``，取平均对数概率最大的类别。
这里只做输入模板封装与前向打分，不改动解码器主干，也没有新增可训练参数。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .classifier import load_lm_checkpoint
from .data import LABELS, ROOT, Example, prepare_data
from .tokenizer import label_verbalizers
from .utils import (Stopwatch, classification_metrics, environment_info,
                    precision_context, seed_everything, write_json)


def truncate(ids, limit):
    return list(ids[:max(0, limit)])


def select_exemplars(examples, tokenizer, per_class=1, max_tokens=24,
                     verbalizer_preset="tuned"):
    """每个类别选一个长度适中的训练样本，避免 prompt 太长或信息太少。"""
    verbalizers = label_verbalizers(verbalizer_preset)
    chosen = {label: [] for label in LABELS}
    for example in examples:
        name = LABELS[example.label]
        if len(chosen[name]) >= per_class:
            continue
        # 把模板和正文一起截断，保证每个示例的 token 数可控。
        tokens = tokenizer.encode(
            f"title: {example.title} description: {example.description}",
            max_length=max_tokens)
        if len(tokens) < 8:
            continue
        chosen[name].append({
            "title": example.title,
            "description": example.description,
            "tokens": tokens,
            "verbalizer": verbalizers[name],
        })
        if all(len(chosen[label]) >= per_class for label in LABELS):
            break
    missing = [label for label in LABELS if len(chosen[label]) < per_class]
    if missing:
        raise RuntimeError(f"训练集中找不到足够的 few-shot 示例：{missing}")
    return chosen


def build_test_prefix(tokenizer, title, description, suffix, budget):
    """构造测试样本前缀，保留末尾 ``topic:``，只截断正文。"""
    body = tokenizer.encode(f"title: {title} description: {description}")
    suffix_ids = tokenizer.encode(suffix)
    if len(suffix_ids) >= budget:
        raise ValueError("上下文预算太小，放不下模板后缀")
    body = truncate(body, budget - len(suffix_ids))
    return body + suffix_ids


def build_shots_prefix(tokenizer, exemplars, per_class, budget):
    """拼接 few-shot 示例；若超长则从最早的示例正文开始截断。"""
    ids = []
    for label in LABELS:
        for exemplar in exemplars[label][:per_class]:
            ids.extend(tokenizer.encode(f"topic: {exemplar['verbalizer']}"))
            # tokens 已经包含 "title: ... description: ..." 模板。
            ids.extend(exemplar["tokens"])
    if len(ids) > budget:
        ids = truncate(ids, budget)
    return ids


def make_sequences(examples, tokenizer, label_ids, shots_prefix, budget,
                   reserve):
    """为每个测试样本生成四类候选序列，并记录标签词元位置。"""
    sequences, positions, owners = [], [], []
    test_budget = budget - reserve - len(shots_prefix)
    if test_budget < 8:
        raise ValueError("few-shot 示例占满上下文，留给测试文本的预算不足")
    for example in examples:
        test_prefix = build_test_prefix(
            tokenizer, example.title, example.description, "topic:", test_budget)
        prefix = list(shots_prefix) + test_prefix
        if len(prefix) + reserve > budget:
            raise AssertionError("few-shot 提示超出上下文预算")
        for class_index, ids in enumerate(label_ids):
            target = list(ids) + [tokenizer.eos_id]
            sequence = prefix + target
            if len(sequence) > budget:
                raise AssertionError("提示构造超出上下文预算")
            sequences.append(sequence)
            start = len(sequence) - len(target)
            positions.append(list(range(start, len(sequence))))
            owners.append(class_index)
    return sequences, positions, owners


@torch.no_grad()
def score_batch(model, tokenizer, sequences, positions, owners, num_classes,
                device, context_length):
    """一次前向给 batch 内所有 (样本, 类别) 序列打分。"""
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_length), tokenizer.pad_id,
                           dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long,
                                                      device=device)
    with precision_context(device):
        hidden = model.encode(input_ids)
    # 只对标签位置接输出头，避免为 [B, L, V] 生成完整 logits。
    row_indices, position_indices, target_tokens = [], [], []
    for row, (target_positions, sequence) in enumerate(zip(positions, sequences)):
        for position in target_positions:
            if position == 0:
                continue
            row_indices.append(row)
            position_indices.append(position - 1)
            target_tokens.append(sequence[position])
    selected = hidden[torch.tensor(row_indices, device=device),
                      torch.tensor(position_indices, device=device)]
    with precision_context(device):
        logits = model.lm_head(selected)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(row_indices), device=device),
        torch.tensor(target_tokens, device=device)]
    scores = torch.zeros(len(sequences), dtype=torch.float32, device=device)
    scores.index_add_(0, torch.tensor(row_indices, device=device), token_log_probs)
    counts = torch.bincount(torch.tensor(row_indices, device=device),
                            minlength=len(sequences)).clamp(min=1)
    scores = (scores / counts).reshape(-1, num_classes)
    return scores.cpu()


@torch.no_grad()
def calibration_scores(model, tokenizer, label_ids, shots_prefix, budget,
                       reserve, device):
    """Contextual calibration：用无内容提示估计并扣除 verbalizer 的先验偏置。

    只使用语言模型，不需要任何标签；三个内容为空的模板取平均，降低单个模板的
    偶然性。校准在测试集之外完成，不构成监督训练。
    """
    probes = [Example("", "", 0), Example("none", "none", 0),
              Example("news", "news", 0)]
    sequences, positions, owners = make_sequences(
        probes, tokenizer, label_ids, shots_prefix, budget, reserve)
    return score_batch(model, tokenizer, sequences, positions, owners,
                       len(label_ids), device, budget).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=0,
                        help="每个类别的示例数；0 为零样本，1 为四样本（四类各一个）")
    parser.add_argument("--verbalizer-preset", choices=["default", "tuned"],
                        default="tuned",
                        help="default 为 world/sports/business/science；tuned 为验证集选出的词")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                        default=True, help="是否做 content-free 上下文校准")
    parser.add_argument("--lm-checkpoint", default=str(ROOT / "results" / "lm_pretrained_best.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=0,
                        help="提示总长度；0 表示使用语言模型 context_length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.shots < 0 or args.batch_size < 1:
        parser.error("shots 不能为负，batch-size 必须为正")
    output = Path(args.output) if args.output else ROOT / "results" / (
        "prompt_zeroshot" if args.shots == 0 else f"prompt_fewshot{args.shots * len(LABELS)}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint, tokenizer, model = load_lm_checkpoint(args.lm_checkpoint, map_location="cpu")
    model = model.to(device).eval()
    context_length = model.cfg.context_length
    max_length = args.max_length or context_length
    if max_length > context_length:
        raise ValueError("提示长度不能超过模型 context_length")
    bundle = prepare_data()
    verbalizers = label_verbalizers(args.verbalizer_preset)
    label_ids = [tokenizer.encode(verbalizers[label]) for label in LABELS]
    if any(len(ids) == 0 for ids in label_ids):
        raise RuntimeError("verbalizer 被编码成空序列")
    reserve = max(len(ids) for ids in label_ids) + 1  # + <eos>
    exemplars = select_exemplars(bundle.train, tokenizer, per_class=args.shots,
                                 verbalizer_preset=args.verbalizer_preset) \
        if args.shots else {label: [] for label in LABELS}
    shots_prefix = build_shots_prefix(tokenizer, exemplars, args.shots,
                                      max_length - reserve) if args.shots else []
    if len(shots_prefix) >= max_length - reserve:
        raise ValueError("few-shot 示例占满了上下文，请减少 shots")
    calibration = calibration_scores(
        model, tokenizer, label_ids, shots_prefix, max_length, reserve,
        device) if args.calibrate else None
    print(f"shots={args.shots} 示例 token 数={len(shots_prefix)} "
          f"上下文={max_length} 标签词元={label_ids} "
          f"校准={args.calibrate}", flush=True)

    examples = bundle.test
    targets = [example.label for example in examples]
    predictions, all_scores = [], []
    inference = Stopwatch(device)
    with inference:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            sequences, positions, owners = make_sequences(
                batch, tokenizer, label_ids, shots_prefix, max_length, reserve)
            scores = score_batch(model, tokenizer, sequences, positions, owners,
                                 len(LABELS), device, context_length)
            if calibration is not None:
                scores = scores - calibration
            all_scores.extend(scores.tolist())
            predictions.extend(scores.argmax(dim=-1).tolist())
            if start % (args.batch_size * 20) == 0:
                print(f"已打分 {min(start + len(batch), len(examples))}/{len(examples)}",
                      flush=True)
    metrics = classification_metrics(targets, predictions, LABELS)
    result = {
        "model": "prompt scoring with frozen decoder LM",
        "shots_per_class": args.shots,
        "num_exemplars": args.shots * len(LABELS),
        "verbalizers": verbalizers,
        "verbalizer_preset": args.verbalizer_preset,
        "calibrated": bool(args.calibrate),
        "calibration_scores": calibration.tolist() if calibration is not None else None,
        "template": "title: {title} description: {description} topic: {verbalizer}<eos>",
        "scoring": "mean log P(verbalizer tokens, <eos> | prompt); no training",
        "trainable_parameters": 0,
        "context_length": context_length,
        "prompt_max_length": max_length,
        "label_ids": label_ids,
        "exemplars": {
            label: [{"title": item["title"], "description": item["description"],
                     "verbalizer": item["verbalizer"]}
                    for item in items]
            for label, items in exemplars.items()
        },
        "test": metrics,
        "inference_seconds": inference.elapsed,
        "test_examples_per_second": len(examples) / inference.elapsed,
        "environment": environment_info(device),
        "data": {
            "counts": bundle.metadata["counts"],
            "train_csv_sha256": bundle.metadata["train_csv_sha256"],
            "test_csv_sha256": bundle.metadata["test_csv_sha256"],
        },
        "lm_checkpoint": str(args.lm_checkpoint),
        "lm_checkpoint_step": checkpoint.get("step"),
    }
    write_json(output / "metrics.json", result)
    write_json(output / "test_predictions.json", {
        "targets": targets,
        "predictions": predictions,
        # 每行四类的平均对数概率，便于后续校准分析。
        "scores": all_scores,
    })
    print(json.dumps({"shots": args.shots, "test": {
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
        "inference_seconds": inference.elapsed,
        "examples_per_second": result["test_examples_per_second"]},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
