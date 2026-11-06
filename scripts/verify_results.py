"""核对数据、日志、checkpoint、预测和报告表格之间的一致性。

不重新训练，只加载已保存的 checkpoint 和预测文件做交叉检查；最后运行一次
单元测试并保存 JUnit XML。任何一项失败都会以非零状态退出。
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import load_classifier_checkpoint, parameter_report
from src.data import LABELS
from src.model import ModelConfig, theoretical_parameters
from src.tokenizer import TUNED_LABEL_VERBALIZERS
from src.utils import write_json


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy(targets, predictions):
    return sum(int(a == b) for a, b in zip(targets, predictions)) / len(targets)


def main():
    results = ROOT / "results"
    data = load("data/metadata.json")
    lm = load("results/lm_summary.json")
    full = load("results/classifier_full/metrics.json")
    frozen = load("results/classifier_frozen/metrics.json")
    scratch = load("results/classifier_scratch/metrics.json")
    nb = load("results/naive_bayes/metrics.json")
    prompt = load("results/prompt_zeroshot/metrics.json")
    prompt_default = load("results/prompt_zeroshot_default/metrics.json")
    fewshot = load("results/prompt_fewshot4/metrics.json")
    tuning = load("results/prompt_verbalizer_tuning.json")
    comparison = load("results/comparison.json")
    samples = load("results/samples.json")
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    # 数据与划分
    check("data_counts", data["counts"] == {"train": 108000, "validation": 12000,
                                             "test": 7600})
    check("train_class_balance", all(value == 27000 for value in
                                     data["class_distribution"]["train"].values()))
    check("validation_class_balance", all(value == 3000 for value in
                                          data["class_distribution"]["validation"].values()))
    check("test_class_balance", all(value == 1900 for value in
                                    data["class_distribution"]["test"].values()))
    check("train_csv_sha256", sha256(ROOT / "data" / "ag_news_train.csv") ==
          data["train_csv_sha256"])
    check("test_csv_sha256", sha256(ROOT / "data" / "ag_news_test.csv") ==
          data["test_csv_sha256"])
    lm_checkpoint = torch.load(ROOT / "results" / "lm_pretrained_best.pt",
                               map_location="cpu", weights_only=False)
    check("vocab_size", data["tokenizer"]["vocab_size"] ==
          len(lm_checkpoint["tokenizer"]["vocabulary"]))

    # 语言模型日志与参数
    config = ModelConfig(**lm["config"])
    check("lm_parameter_formula", lm["parameters"]["total"] ==
          theoretical_parameters(config))
    check("lm_loss_decreases", lm["final"]["train_loss"] < lm["initial"]["train_loss"]
          and lm["final"]["validation_loss"] < lm["initial"]["validation_loss"])
    with (results / "lm_training_log.csv").open() as handle:
        training = list(csv.DictReader(handle))
    with (results / "lm_eval_log.csv").open() as handle:
        evaluation = list(csv.DictReader(handle))
    check("lm_2000_steps", [int(row["step"]) for row in training] == list(range(1, 2001)))
    check("lm_training_log_finite", all(
        math.isfinite(float(row["loss"])) and
        math.isfinite(float(row["gradient_norm_before_clip"])) for row in training))
    check("lm_eval_steps", [int(row["step"]) for row in evaluation] ==
          list(range(0, 2001, 250)))
    check("lm_summary_matches_eval", all(
        float(evaluation[-1][key]) == lm["final"][key]
        for key in ("train_loss", "validation_loss", "validation_perplexity")))
    check("lm_checkpoint_step", lm_checkpoint["step"] == lm["settings"]["steps"])
    check("lm_checkpoint_vocab", lm_checkpoint["config"]["vocab_size"] ==
          config.vocab_size)

    # 分类器 checkpoint、预测与指标
    for name, metrics in (("full", full), ("frozen", frozen), ("scratch", scratch)):
        model, tokenizer, payload = load_classifier_checkpoint(
            ROOT / "results" / f"classifier_{name}" / "best.pt")
        report = parameter_report(model)
        check(f"classifier_{name}_parameter_report",
              report["total"] == metrics["parameters"]["total"] and
              report["trainable"] == metrics["parameters"]["trainable"])
        predictions = load(f"results/classifier_{name}/test_predictions.json")
        check(f"classifier_{name}_prediction_count", len(predictions["predictions"]) == 7600)
        check(f"classifier_{name}_accuracy_matches_predictions",
              abs(accuracy(predictions["targets"], predictions["predictions"]) -
                  metrics["test"]["accuracy"]) < 1e-12)
        check(f"classifier_{name}_confusion_sum",
              sum(sum(row) for row in metrics["test"]["confusion_matrix"]) == 7600)
        check(f"classifier_{name}_tokenizer_size", len(tokenizer) == config.vocab_size)
    check("full_trainable_equals_total",
          full["parameters"]["trainable"] == full["parameters"]["total"])
    check("frozen_only_head_trainable",
          frozen["parameters"]["trainable"] == frozen["parameters"]["classification_head"])
    check("scratch_differs_from_pretrained", scratch["test"]["accuracy"] !=
          full["test"]["accuracy"])
    check("full_beats_scratch", full["test"]["accuracy"] > scratch["test"]["accuracy"])
    check("scratch_beats_frozen", scratch["test"]["accuracy"] > frozen["test"]["accuracy"])

    # 朴素贝叶斯
    check("nb_alpha_selected", nb["alpha"] in nb["alpha_candidates"])
    check("nb_parameter_count", nb["parameters"]["total"] ==
          nb["feature_config"]["n_features"] * len(LABELS) + len(LABELS))
    check("nb_sklearn_agreement", nb["sklearn_crosscheck"]["prediction_agreement"] == 1.0)
    check("nb_sklearn_accuracy_match",
          nb["sklearn_crosscheck"]["accuracy_difference"] < 1e-12)
    nb_predictions = load("results/naive_bayes/test_predictions.json")
    check("nb_prediction_count", len(nb_predictions["predictions"]) == 7600)
    check("nb_accuracy_matches_predictions",
          abs(accuracy(nb_predictions["targets"], nb_predictions["predictions"]) -
              nb["test"]["accuracy"]) < 1e-12)

    # Prompt
    check("prompt_tuning_validation_only", tuning["test_set_used"] is False)
    check("prompt_tuning_selected_mapping",
          tuning["selected"] == TUNED_LABEL_VERBALIZERS)
    check("prompt_uses_tuned_mapping", prompt["verbalizers"] ==
          TUNED_LABEL_VERBALIZERS)
    check("prompt_is_calibrated", prompt["calibrated"] is True)
    check("prompt_no_trainable_parameters", prompt["trainable_parameters"] == 0)
    check("prompt_fewshot_differs_from_zero",
          fewshot["test"]["accuracy"] != prompt["test"]["accuracy"])
    check("prompt_default_differs_from_tuned",
          prompt_default["test"]["accuracy"] != prompt["test"]["accuracy"])
    check("prompt_accuracy_matches_predictions",
          abs(accuracy(load("results/prompt_zeroshot/test_predictions.json")["targets"],
                       load("results/prompt_zeroshot/test_predictions.json")["predictions"]) -
              prompt["test"]["accuracy"]) < 1e-12)

    # 样例与汇总表
    check("three_samples", len(samples) == 3)
    for row in samples:
        for name in ("decoder_full", "decoder_frozen", "decoder_scratch",
                     "naive_bayes", "prompt_zeroshot", "prompt_fewshot4"):
            source = {
                "decoder_full": "classifier_full", "decoder_frozen": "classifier_frozen",
                "decoder_scratch": "classifier_scratch", "naive_bayes": "naive_bayes",
                "prompt_zeroshot": "prompt_zeroshot",
                "prompt_fewshot4": "prompt_fewshot4",
            }[name]
            stored = load(f"results/{source}/test_predictions.json")["predictions"][row["index"]]
            check(f"sample_{row['index']}_{name}_matches",
                  LABELS[stored] == row["predictions"][name])
    check("comparison_seven_rows", len(comparison["rows"]) == 7)
    source_metrics = {
        "decoder_full": full, "decoder_frozen": frozen, "decoder_scratch": scratch,
        "naive_bayes": nb, "prompt_zeroshot": prompt,
        "prompt_zeroshot_default": prompt_default, "prompt_fewshot4": fewshot,
    }
    check("comparison_matches_metrics", all(
        abs(row["test_accuracy"] - source_metrics[row["key"]]["test"]["accuracy"]) < 1e-12
        for row in comparison["rows"]))

    # 源码约束
    model_ast = ast.parse((ROOT / "src" / "model.py").read_text(encoding="utf-8"))
    prohibited = {"MultiheadAttention", "Transformer", "TransformerEncoder",
                  "TransformerDecoder", "scaled_dot_product_attention",
                  "LayerNorm", "layer_norm"}
    called = {node.func.attr for node in ast.walk(model_ast)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    check("model_avoids_high_level_layers", not prohibited.intersection(called))
    classifier_source = (ROOT / "src" / "classifier.py").read_text(encoding="utf-8")
    check("classifier_head_present", "classification_head" in classifier_source)
    check("classifier_uses_encode", "backbone.encode" in classifier_source)

    # 单元测试；保存机器可读和人类可读日志。
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--junitxml=results/tests.xml"],
        cwd=ROOT, text=True, capture_output=True)
    (results / "tests_console.log").write_text(test.stdout + test.stderr,
                                               encoding="utf-8")
    check("all_tests_pass", test.returncode == 0 and "passed" in test.stdout)

    report = {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "dataset": data["source"],
        "test_counts": data["counts"],
        "accuracies": {
            "decoder_full": full["test"]["accuracy"],
            "decoder_frozen": frozen["test"]["accuracy"],
            "decoder_scratch": scratch["test"]["accuracy"],
            "naive_bayes": nb["test"]["accuracy"],
            "prompt_zeroshot": prompt["test"]["accuracy"],
            "prompt_fewshot4": fewshot["test"]["accuracy"],
        },
    }
    write_json(results / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit("一致性验证失败")


if __name__ == "__main__":
    main()
