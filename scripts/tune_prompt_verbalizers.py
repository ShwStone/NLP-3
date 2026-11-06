"""仅在验证集上选择 Prompt verbalizer，不接触测试集。

零样本 Prompt 的类别词选择属于超参数；本脚本在验证集前 4,000 条上枚举候选
组合，再用完整验证集复核最优组合。测试集由 ``src.prompt`` 单独评估一次。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import load_lm_checkpoint
from src.data import LABELS, prepare_data
from src.prompt import calibration_scores, make_sequences, score_batch
from src.utils import seed_everything, write_json

CANDIDATES = {
    "World": ["world", "international", "politics"],
    "Sports": ["sports", "sport"],
    "Business": ["business", "economy", "markets", "financial"],
    "Sci/Tech": ["science", "technology", "computers"],
}
SCREEN_SIZE = 4000


@torch.no_grad()
def evaluate(model, tokenizer, examples, label_words, device, context_length=192):
    label_ids = [tokenizer.encode(word) for word in label_words]
    reserve = max(len(ids) for ids in label_ids) + 1
    calibration = calibration_scores(
        model, tokenizer, label_ids, [], context_length, reserve, device)
    predictions, targets = [], [example.label for example in examples]
    for start in range(0, len(examples), 64):
        batch = examples[start:start + 64]
        sequences, positions, owners = make_sequences(
            batch, tokenizer, label_ids, [], context_length, reserve)
        scores = score_batch(model, tokenizer, sequences, positions, owners,
                             len(LABELS), device, context_length) - calibration
        predictions.extend(scores.argmax(dim=-1).tolist())
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_data()
    checkpoint, tokenizer, model = load_lm_checkpoint(
        ROOT / "results" / "lm_pretrained_best.pt", map_location="cpu")
    model = model.to(device).eval()
    screening = bundle.validation[:SCREEN_SIZE]
    results = []
    for combination in itertools.product(*[CANDIDATES[label] for label in LABELS]):
        accuracy = evaluate(model, tokenizer, screening, combination, device)
        results.append({"verbalizers": dict(zip(LABELS, combination)),
                        "screening_accuracy": accuracy})
        print(f"{combination} -> {accuracy:.4f}", flush=True)
    results.sort(key=lambda row: row["screening_accuracy"], reverse=True)
    best_words = tuple(results[0]["verbalizers"][label] for label in LABELS)
    # 用完整验证集复核选中的组合，仍然不看测试集。
    full_validation_accuracy = evaluate(
        model, tokenizer, bundle.validation, best_words, device)
    report = {
        "selection_data": "validation split only; screening on first 4,000 examples, "
                          "then full 12,000-example validation check",
        "screening_size": SCREEN_SIZE,
        "candidates": CANDIDATES,
        "num_combinations": len(results),
        "results_sorted": results,
        "selected": dict(zip(LABELS, best_words)),
        "selected_screening_accuracy": results[0]["screening_accuracy"],
        "selected_full_validation_accuracy": full_validation_accuracy,
        "test_set_used": False,
    }
    write_json(ROOT / "results" / "prompt_verbalizer_tuning.json", report)
    print(json.dumps({"selected": report["selected"],
                      "screening_accuracy": report["selected_screening_accuracy"],
                      "full_validation_accuracy": full_validation_accuracy},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
