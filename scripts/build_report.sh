#!/usr/bin/env bash
# 连续运行两次 XeLaTeX，生成目录、交叉引用和最终 PDF。
set -euo pipefail
cd "$(dirname "$0")/../report"
xelatex -interaction=nonstopmode -halt-on-error report.tex > build_pass_1.log
xelatex -interaction=nonstopmode -halt-on-error report.tex > build_pass_2.log
cp report.pdf ../实验报告_文本分类.pdf
echo "报告已编译：report/report.pdf 与 实验报告_文本分类.pdf"
