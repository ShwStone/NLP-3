"""汇总三种方案的测试指标、参数量和耗时，生成报告使用的对比表。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def classifier_row(key, title, path):
    metrics = load(path)
    return {
        "key": key,
        "method": title,
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "validation_accuracy": metrics["best_validation"]["accuracy"],
        "trainable_parameters": metrics["parameters"]["trainable"],
        "total_parameters": metrics["parameters"]["total"],
        "backbone_parameters": metrics["parameters"]["backbone"],
        "head_parameters": metrics["parameters"]["classification_head"],
        "training_seconds": metrics["training_wall_seconds"],
        "inference_seconds": metrics["inference_wall_seconds"],
        "test_examples_per_second": metrics["test_examples_per_second"],
        "epochs": metrics["settings"]["epochs"],
        "best_epoch": metrics["best_epoch"],
        "peak_cuda_mib": (metrics.get("peak_cuda_allocated_bytes") or 0) / 2 ** 20,
        "checkpoint_mib": Path(ROOT / metrics["checkpoint"]).stat().st_size / 2 ** 20,
        "notes": metrics["settings"]["mode"],
    }


def main():
    rows = [
        classifier_row("decoder_full", "解码器全参数微调", "results/classifier_full/metrics.json"),
        classifier_row("decoder_frozen", "解码器冻结主干（线性探针）", "results/classifier_frozen/metrics.json"),
        classifier_row("decoder_scratch", "解码器随机初始化（消融）", "results/classifier_scratch/metrics.json"),
    ]
    nb = load("results/naive_bayes/metrics.json")
    rows.append({
        "key": "naive_bayes",
        "method": "多项式朴素贝叶斯",
        "test_accuracy": nb["test"]["accuracy"],
        "test_macro_f1": nb["test"]["macro_f1"],
        "validation_accuracy": nb["validation"]["accuracy"],
        "trainable_parameters": nb["parameters"]["total"],
        "total_parameters": nb["parameters"]["total"],
        "backbone_parameters": 0,
        "head_parameters": nb["parameters"]["total"],
        "training_seconds": nb["train_seconds"]["total"],
        "inference_seconds": nb["inference_seconds_including_transform"],
        "test_examples_per_second": nb["test_examples_per_second"],
        "epochs": 1,
        "best_epoch": 1,
        "peak_cuda_mib": 0.0,
        "checkpoint_mib": 0.0,
        "notes": f"alpha={nb['alpha']}, features={nb['feature_config']['n_features']}",
    })
    prompt = load("results/prompt_zeroshot/metrics.json")
    rows.append({
        "key": "prompt_zeroshot",
        "method": "Prompt 零样本（校准）",
        "test_accuracy": prompt["test"]["accuracy"],
        "test_macro_f1": prompt["test"]["macro_f1"],
        "validation_accuracy": None,
        "trainable_parameters": 0,
        "total_parameters": 0,
        "backbone_parameters": 10888704,
        "head_parameters": 0,
        "training_seconds": 0.0,
        "inference_seconds": prompt["inference_seconds"],
        "test_examples_per_second": prompt["test_examples_per_second"],
        "epochs": 0,
        "best_epoch": 0,
        "peak_cuda_mib": 0.0,
        "checkpoint_mib": 0.0,
        "notes": "tuned verbalizers; contextual calibration; no training",
    })
    prompt_default = load("results/prompt_zeroshot_default/metrics.json")
    rows.append({
        "key": "prompt_zeroshot_default",
        "method": "Prompt 零样本（默认词）",
        "test_accuracy": prompt_default["test"]["accuracy"],
        "test_macro_f1": prompt_default["test"]["macro_f1"],
        "validation_accuracy": None,
        "trainable_parameters": 0,
        "total_parameters": 0,
        "backbone_parameters": 10888704,
        "head_parameters": 0,
        "training_seconds": 0.0,
        "inference_seconds": prompt_default["inference_seconds"],
        "test_examples_per_second": prompt_default["test_examples_per_second"],
        "epochs": 0,
        "best_epoch": 0,
        "peak_cuda_mib": 0.0,
        "checkpoint_mib": 0.0,
        "notes": "world/sports/business/science; no calibration",
    })
    fewshot = load("results/prompt_fewshot4/metrics.json")
    rows.append({
        "key": "prompt_fewshot4",
        "method": "Prompt 四样本（校准）",
        "test_accuracy": fewshot["test"]["accuracy"],
        "test_macro_f1": fewshot["test"]["macro_f1"],
        "validation_accuracy": None,
        "trainable_parameters": 0,
        "total_parameters": 0,
        "backbone_parameters": 10888704,
        "head_parameters": 0,
        "training_seconds": 0.0,
        "inference_seconds": fewshot["inference_seconds"],
        "test_examples_per_second": fewshot["test_examples_per_second"],
        "epochs": 0,
        "best_epoch": 0,
        "peak_cuda_mib": 0.0,
        "checkpoint_mib": 0.0,
        "notes": "one exemplar per class; contextual calibration; no training",
    })
    comparison = {
        "dataset": "AG News",
        "test_examples": 7600,
        "rows": rows,
        "ranking_by_accuracy": [
            row["key"] for row in sorted(rows, key=lambda item: item["test_accuracy"],
                                        reverse=True)
        ],
    }
    output = ROOT / "results" / "comparison.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    with (ROOT / "results" / "comparison.csv").open("w", newline="",
                                                     encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']:32s} acc={row['test_accuracy']:.4f} "
              f"f1={row['test_macro_f1']:.4f} trainable={row['trainable_parameters']:,} "
              f"train={row['training_seconds']:.2f}s")
    print(f"已写出 {output}")


if __name__ == "__main__":
    main()
