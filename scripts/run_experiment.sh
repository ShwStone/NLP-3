#!/usr/bin/env bash
# 一键复现作业三：数据 -> 预训练 -> 三种分类器 -> NB -> Prompt ->
# 汇总/图表/报告 -> 测试/核验/打包。
#
# 注意：这会覆盖 results/ 中的训练日志、指标和报告表格。请先确认已有结果
# 已经备份。PYTHON 可指定解释器，例如 PYTHON=/path/to/python。
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p results

echo "== 1/12 数据准备 =="
"$PYTHON" -m src.data

echo "== 2/12 语言模型预训练 =="
"$PYTHON" -m src.pretrain --steps 2000 --batch-size 64 --eval-interval 250

echo "== 3/12 验证集 verbalizer 搜索（不接触测试集）=="
"$PYTHON" scripts/tune_prompt_verbalizers.py

echo "== 4/12 全参数微调 =="
"$PYTHON" -m src.train_classifier --mode full --epochs 3

echo "== 5/12 冻结主干 =="
"$PYTHON" -m src.train_classifier --mode frozen --epochs 6

echo "== 6/12 随机初始化消融 =="
"$PYTHON" -m src.train_classifier --mode scratch --epochs 3

echo "== 7/12 朴素贝叶斯 =="
"$PYTHON" -m src.naive_bayes

echo "== 8/12 Prompt 实验 =="
"$PYTHON" -m src.prompt --shots 0 --verbalizer-preset tuned \
  --output results/prompt_zeroshot
"$PYTHON" -m src.prompt --shots 0 --verbalizer-preset default --no-calibrate \
  --output results/prompt_zeroshot_default
"$PYTHON" -m src.prompt --shots 1 --verbalizer-preset tuned \
  --output results/prompt_fewshot4

echo "== 9/12 样例与汇总 =="
"$PYTHON" -m src.make_samples
"$PYTHON" scripts/collect_results.py
"$PYTHON" scripts/make_figures.py
"$PYTHON" scripts/report_values.py

echo "== 10/12 单元测试与一致性核验 =="
"$PYTHON" scripts/verify_results.py

echo "== 11/12 编译中文报告 =="
if command -v xelatex >/dev/null 2>&1; then
  bash scripts/build_report.sh
else
  echo "未找到 xelatex，跳过报告编译；已生成 report/*.tex。"
fi

echo "== 12/12 提交检查与打包 =="
"$PYTHON" scripts/check_delivery.py
"$PYTHON" scripts/package_submission.py
echo "全部完成。"
