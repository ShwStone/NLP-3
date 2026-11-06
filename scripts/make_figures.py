"""根据真实日志生成报告图表（PDF 供 LaTeX，PNG 供预览）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 140,
})


def save(figure, name):
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight",
                       dpi=300 if suffix == "png" else None)
    plt.close(figure)


def lm_curve():
    with (ROOT / "results" / "lm_training_log.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    with (ROOT / "results" / "lm_eval_log.csv").open() as handle:
        evaluation = list(csv.DictReader(handle))
    steps = np.array([int(row["step"]) for row in raw])
    losses = np.array([float(row["loss"]) for row in raw])
    eval_steps = [int(row["step"]) for row in evaluation]
    eval_train = [float(row["train_loss"]) for row in evaluation]
    eval_val = [float(row["validation_loss"]) for row in evaluation]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    axes[0].plot(steps, losses, color="#9fb3c8", linewidth=0.8, label="per-step training loss")
    axes[0].plot(eval_steps, eval_train, color="#173B59", marker="o", markersize=3,
                 linewidth=1.4, label="fixed-window train")
    axes[0].plot(eval_steps, eval_val, color="#C0504D", marker="s", markersize=3,
                 linewidth=1.4, label="fixed-window validation")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("cross-entropy (nats/token)")
    axes[0].set_title("Language-model pretraining")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(eval_steps, [np.exp(value) for value in eval_train],
                 color="#173B59", marker="o", markersize=3, label="train")
    axes[1].plot(eval_steps, [np.exp(value) for value in eval_val],
                 color="#C0504D", marker="s", markersize=3, label="validation")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("Perplexity (log scale)")
    axes[1].legend(frameon=False, fontsize=7)
    save(figure, "lm_pretraining")


def classifier_curves():
    modes = [
        ("full", "Full fine-tune", "#173B59", "o"),
        ("frozen", "Frozen backbone", "#2E7D6E", "s"),
        ("scratch", "Random init", "#C0504D", "^"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for mode, label, color, marker in modes:
        metrics = json.loads((ROOT / "results" / f"classifier_{mode}" / "metrics.json")
                             .read_text())
        history = metrics["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=color,
                     marker=marker, markersize=3, linewidth=1.4, label=label)
        axes[1].plot(epochs, [100 * row["validation_accuracy"] for row in history],
                     color=color, marker=marker, markersize=3, linewidth=1.4, label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training cross-entropy")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_title("Validation accuracy")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    save(figure, "classifier_curves")


def comparison_chart():
    comparison = json.loads((ROOT / "results" / "comparison.json").read_text())
    keys = ["decoder_full", "decoder_scratch", "naive_bayes",
            "decoder_frozen", "prompt_zeroshot", "prompt_fewshot4"]
    rows = {row["key"]: row for row in comparison["rows"]}
    labels = ["Decoder\nfull FT", "Decoder\nscratch", "Naive\nBayes",
              "Decoder\nfrozen", "Prompt\n0-shot", "Prompt\n4-shot"]
    values = [100 * rows[key]["test_accuracy"] for key in keys]
    colors = ["#173B59", "#6C8EBF", "#C9A227", "#2E7D6E", "#C0504D", "#A56E9B"]
    figure, axis = plt.subplots(figsize=(6.4, 2.9))
    bars = axis.bar(labels, values, color=colors, width=0.68)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}",
                  ha="center", va="bottom", fontsize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("test accuracy (%)")
    axis.set_title("AG News test accuracy")
    axis.grid(axis="x", visible=False)
    save(figure, "comparison")


def confusion_full():
    metrics = json.loads((ROOT / "results" / "classifier_full" / "metrics.json")
                         .read_text())
    matrix = np.array(metrics["test"]["confusion_matrix"])
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(4), labels, rotation=25, ha="right")
    axis.set_yticks(range(4), labels)
    axis.set_xlabel("predicted")
    axis.set_ylabel("gold")
    axis.set_title("Full fine-tune confusion matrix")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center",
                      va="center", fontsize=8,
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.grid(False)
    save(figure, "confusion_full")


def main():
    lm_curve()
    classifier_curves()
    comparison_chart()
    confusion_full()
    print("图表已生成：", sorted(path.name for path in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
