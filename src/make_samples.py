"""导出三组固定测试样本，打印所有模型的预测，供报告样例表使用。

样本刻意覆盖三种典型情况：
  * 索引 37：World，零样本 Prompt 判成 Business；
  * 索引 41：Business，NB 判成 Sci/Tech；
  * 索引 397：Sci/Tech，全参数微调判成 Business，NB/Prompt 判对。
索引基于固定数据划分，重跑实验不会改变。
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import LABELS, ROOT, prepare_data
from .utils import write_json

SAMPLE_INDICES = [37, 41, 397]
MODEL_PATHS = {
    "decoder_full": "classifier_full",
    "decoder_frozen": "classifier_frozen",
    "decoder_scratch": "classifier_scratch",
    "naive_bayes": "naive_bayes",
    "prompt_zeroshot": "prompt_zeroshot",
    "prompt_fewshot4": "prompt_fewshot4",
}


def main():
    bundle = prepare_data()
    predictions = {}
    for name, directory in MODEL_PATHS.items():
        path = ROOT / "results" / directory / "test_predictions.json"
        predictions[name] = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    samples = []
    for index in SAMPLE_INDICES:
        example = bundle.test[index]
        row = {
            "index": index,
            "gold": LABELS[example.label],
            "title": example.title,
            "description": example.description,
            "text": example.text,
            "predictions": {
                name: LABELS[values[index]] for name, values in predictions.items()
            },
            "correct": {
                name: values[index] == example.label
                for name, values in predictions.items()
            },
        }
        samples.append(row)
    write_json(ROOT / "results" / "samples.json", samples)
    lines = []
    for row in samples:
        lines.append(f"=== 测试样本 index={row['index']}，金标准 {row['gold']} ===")
        lines.append(f"Title: {row['title']}")
        lines.append(f"Description: {row['description']}")
        lines.append("预测：")
        for name, value in row["predictions"].items():
            mark = "✓" if row["correct"][name] else "✗"
            lines.append(f"  {name:18s} {value:9s} {mark}")
        lines.append("")
    text = "\n".join(lines)
    (ROOT / "results" / "samples.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
