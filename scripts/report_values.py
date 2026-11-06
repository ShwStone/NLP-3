"""从真实 JSON 结果生成 LaTeX 宏和表格行，避免手抄实验数字。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def num(value, digits=2):
    return f"{value:,.{digits}f}"


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}"


def tex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\^{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def main():
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
    rows = {row["key"]: row for row in comparison["rows"]}

    values = {
        "DatasetTrain": f"{data['counts']['train']:,}",
        "DatasetValidation": f"{data['counts']['validation']:,}",
        "DatasetTest": f"{data['counts']['test']:,}",
        "VocabSize": f"{data['tokenizer']['vocab_size']:,}",
        "MeanTokens": f"{data['token_lengths']['train']['mean_tokens']:.2f}",
        "MedianTokens": f"{data['token_lengths']['train']['median_tokens']:.0f}",
        "TokenPnn": f"{data['token_lengths']['train']['p99_tokens']:.0f}",
        "MaxTokens": f"{data['token_lengths']['train']['max_tokens']:.0f}",
        "TruncatedTrainPct": f"{100 * data['token_lengths']['train']['truncated_fraction']:.3f}",
        "TruncatedTestPct": f"{100 * data['token_lengths']['test']['truncated_fraction']:.3f}",
        "LMParams": f"{lm['parameters']['total']:,}",
        "LMInitLoss": num(lm["initial"]["train_loss"], 4),
        "LMFinalLoss": num(lm["final"]["train_loss"], 4),
        "LMInitVal": num(lm["initial"]["validation_loss"], 4),
        "LMFinalVal": num(lm["final"]["validation_loss"], 4),
        "LMInitPpl": num(lm["initial"]["validation_perplexity"]),
        "LMFinalPpl": num(lm["final"]["validation_perplexity"]),
        "LMTrainSeconds": num(lm["training_wall_seconds_including_eval"]),
        "LMTokens": f"{lm['processed_training_tokens']:,}",
        "LMStreamTokens": f"{lm['train_stream_tokens']:,}",
        "LMPeakMiB": num(lm["peak_cuda_allocated_bytes"] / 2 ** 20, 0),
        "FullAcc": pct(full["test"]["accuracy"]),
        "FullFOne": pct(full["test"]["macro_f1"]),
        "FullValAcc": pct(full["best_validation"]["accuracy"]),
        "FullTrainable": f"{full['parameters']['trainable']:,}",
        "FullTotal": f"{full['parameters']['total']:,}",
        "FullBackbone": f"{full['parameters']['backbone']:,}",
        "FullHead": f"{full['parameters']['classification_head']:,}",
        "FullTrainSeconds": num(full["training_wall_seconds"]),
        "FullInferSeconds": num(full["inference_wall_seconds"]),
        "FullThroughput": f"{full['test_examples_per_second']:,.0f}",
        "FullBestEpoch": str(full["best_epoch"]),
        "FullPeakMiB": num((full.get("peak_cuda_allocated_bytes") or 0) / 2 ** 20, 0),
        "FrozenAcc": pct(frozen["test"]["accuracy"]),
        "FrozenFOne": pct(frozen["test"]["macro_f1"]),
        "FrozenTrainable": f"{frozen['parameters']['trainable']:,}",
        "FrozenTrainSeconds": num(frozen["training_wall_seconds"]),
        "FrozenThroughput": f"{frozen['test_examples_per_second']:,.0f}",
        "FrozenBestEpoch": str(frozen["best_epoch"]),
        "ScratchAcc": pct(scratch["test"]["accuracy"]),
        "ScratchFOne": pct(scratch["test"]["macro_f1"]),
        "ScratchTrainSeconds": num(scratch["training_wall_seconds"]),
        "ScratchThroughput": f"{scratch['test_examples_per_second']:,.0f}",
        "NBAcc": pct(nb["test"]["accuracy"]),
        "NBFOne": pct(nb["test"]["macro_f1"]),
        "NBValAcc": pct(nb["validation"]["accuracy"]),
        "NBAlpha": f"{nb['alpha']:g}",
        "NBFeatures": f"{nb['feature_config']['n_features']:,}",
        "NBParams": f"{nb['parameters']['total']:,}",
        "NBTrainSeconds": num(nb["train_seconds"]["total"]),
        "NBFeatureSeconds": num(nb["train_seconds"]["feature_extraction_and_transform"]),
        "NBInferSeconds": num(nb["inference_seconds_including_transform"], 4),
        "NBThroughput": f"{nb['test_examples_per_second']:,.0f}",
        "NBCrossAgreement": pct(nb["sklearn_crosscheck"]["prediction_agreement"], 1),
        "PromptAcc": pct(prompt["test"]["accuracy"]),
        "PromptFOne": pct(prompt["test"]["macro_f1"]),
        "PromptDefaultAcc": pct(prompt_default["test"]["accuracy"]),
        "PromptFewAcc": pct(fewshot["test"]["accuracy"]),
        "PromptFewFOne": pct(fewshot["test"]["macro_f1"]),
        "PromptInferSeconds": num(prompt["inference_seconds"], 2),
        "PromptThroughput": f"{prompt['test_examples_per_second']:,.0f}",
        "PromptTunedVal": pct(tuning["selected_full_validation_accuracy"]),
        "PromptTuningCombos": str(tuning["num_combinations"]),
        "PromptShotsTokens": str(108),
        "ComparisonBest": rows[comparison["ranking_by_accuracy"][0]]["method"],
        "SampleCount": str(len(samples)),
        "Seed": str(data["split_seed"]),
        "TorchVersion": full["environment"]["torch"],
        "PythonVersion": full["environment"]["python"],
        "GPU": full["environment"]["gpu"],
        "CUDAVersion": str(full["environment"]["cuda_runtime"]),
        "FullLoss": num(full["test"]["loss"], 4),
        "PromptSelectedWorld": prompt["verbalizers"]["World"],
        "PromptSelectedSports": prompt["verbalizers"]["Sports"],
        "PromptSelectedBusiness": prompt["verbalizers"]["Business"],
        "PromptSelectedSciTech": prompt["verbalizers"]["Sci/Tech"],
    }
    lines = ["% 本文件由 scripts/report_values.py 自动生成，请勿手工修改。"]
    for key, value in values.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    (REPORT / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 主对比表：准确率、宏 F1、可训练参数、总/固定参数、训练时间、推理吞吐。
    comparison_lines = []
    order = ["decoder_full", "decoder_scratch", "naive_bayes",
             "decoder_frozen", "prompt_zeroshot", "prompt_fewshot4"]
    short_names = {
        "decoder_full": "全参数微调",
        "decoder_scratch": "随机初始化",
        "naive_bayes": "朴素贝叶斯",
        "decoder_frozen": "冻结主干",
        "prompt_zeroshot": "Prompt 零样本",
        "prompt_fewshot4": "Prompt 四样本",
    }
    for key in order:
        row = rows[key]
        if key.startswith("prompt"):
            params = "0（冻结 10.89M）"
            train = "0"
        else:
            params = f"{row['trainable_parameters']:,}"
            train = f"{row['training_seconds']:.2f}"
        comparison_lines.append(
            f"{short_names[key]} & {pct(row['test_accuracy'])} & {pct(row['test_macro_f1'])} & "
            f"{params} & {train} & {row['test_examples_per_second']:,.0f} \\\\")
    (REPORT / "comparison_rows.tex").write_text(
        "\n".join(comparison_lines) + "\n\\bottomrule\n", encoding="utf-8")

    history_lines = []
    for mode, label in (("full", "全参数微调"), ("frozen", "冻结主干"),
                        ("scratch", "随机初始化")):
        metrics = {"full": full, "frozen": frozen, "scratch": scratch}[mode]
        for record in metrics["history"]:
            history_lines.append(
                f"{label} & {record['epoch']} & {record['train_loss']:.4f} & "
                f"{record['validation_loss']:.4f} & "
                f"{100 * record['validation_accuracy']:.2f} \\\\")
    (REPORT / "history_rows.tex").write_text(
        "\n".join(history_lines) + "\n\\bottomrule\n", encoding="utf-8")

    parameter_lines = []
    for label, metrics in (("全参数微调", full), ("冻结主干", frozen),
                           ("随机初始化", scratch)):
        parameter_lines.append(
            f"{label} & {metrics['parameters']['backbone']:,} & "
            f"{metrics['parameters']['classification_head']:,} & "
            f"{metrics['parameters']['trainable']:,} & "
            f"{metrics['parameters']['total']:,} \\\\")
    parameter_lines.append(
        f"朴素贝叶斯 & 0 & {nb['parameters']['total']:,} & "
        f"{nb['parameters']['total']:,} & {nb['parameters']['total']:,} \\\\")
    parameter_lines.append(
        "Prompt（冻结主干） & 10,888,704 & 0 & 0 & 10,888,704 \\\\")
    (REPORT / "parameter_rows.tex").write_text(
        "\n".join(parameter_lines) + "\n\\bottomrule\n", encoding="utf-8")

    per_class_lines = []
    for name, metrics in full["test"]["per_class"].items():
        per_class_lines.append(
            f"{name} & {pct(metrics['precision'])} & {pct(metrics['recall'])} & "
            f"{pct(metrics['f1'])} & {metrics['support']:,} \\\\")
    (REPORT / "per_class_rows.tex").write_text(
        "\n".join(per_class_lines) + "\n\\bottomrule\n", encoding="utf-8")

    top_lines = []
    for label in ("World", "Sports", "Business", "Sci/Tech"):
        features = nb["top_features"][label][:6]
        text = ", ".join(tex_escape(item["feature"]) for item in features)
        top_lines.append(f"{label} & {text} \\\\")
    (REPORT / "nb_top_rows.tex").write_text(
        "\n".join(top_lines) + "\n\\bottomrule\n", encoding="utf-8")

    tuning_lines = []
    for row in tuning["results_sorted"][:5]:
        words = ", ".join(row["verbalizers"][label] for label in
                          ("World", "Sports", "Business", "Sci/Tech"))
        tuning_lines.append(
            f"{words} & {pct(row['screening_accuracy'])} \\\\")
    (REPORT / "tuning_rows.tex").write_text(
        "\n".join(tuning_lines) + "\n\\bottomrule\n", encoding="utf-8")

    sample_lines = []
    for row in samples:
        predictions = row["predictions"]
        sample_lines.append(
            f"{row['index']} & {row['gold']} & {predictions['decoder_full']} & "
            f"{predictions['naive_bayes']} & {predictions['prompt_zeroshot']} & "
            f"{predictions['decoder_frozen']} & {predictions['prompt_fewshot4']} \\\\")
    (REPORT / "sample_rows.tex").write_text(
        "\n".join(sample_lines) + "\n\\bottomrule\n", encoding="utf-8")
    print("已生成 report/values.tex 与报告表格行。")


if __name__ == "__main__":
    main()
