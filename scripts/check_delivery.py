"""提交前检查：文件齐全、报告可打开、指标有限、验证与测试通过。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "requirements.txt", "pytest.ini",
    "src/model.py", "src/tokenizer.py", "src/data.py", "src/classifier.py",
    "src/pretrain.py", "src/train_classifier.py", "src/naive_bayes.py",
    "src/prompt.py", "src/make_samples.py", "src/utils.py",
    "tests/test_model.py", "tests/test_tokenizer.py", "tests/test_classifier.py",
    "tests/test_naive_bayes.py", "tests/test_prompt.py", "tests/test_data.py",
    "report/report.tex", "report/report.pdf", "report/values.tex",
    "实验报告_文本分类.pdf",
    "results/verification.json", "results/comparison.json", "results/comparison.csv",
    "results/samples.json", "results/samples.txt", "results/tests.xml",
    "results/classifier_full/metrics.json", "results/classifier_frozen/metrics.json",
    "results/classifier_scratch/metrics.json", "results/naive_bayes/metrics.json",
    "results/prompt_zeroshot/metrics.json", "results/prompt_fewshot4/metrics.json",
    "data/metadata.json",
]


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main():
    checks = {}
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    checks["all_required_files_exist"] = not missing
    checks["missing_files"] = missing
    checks["verification_passed"] = json.loads(
        (ROOT / "results" / "verification.json").read_text())["all_passed"]
    checks["tests_passed"] = "passed" in (
        ROOT / "results" / "tests_console.log").read_text()
    checks["report_pdf_nonempty"] = (ROOT / "report" / "report.pdf").stat().st_size > 100_000
    checks["report_copy_nonempty"] = (ROOT / "实验报告_文本分类.pdf").stat().st_size > 100_000
    checks["report_tex_no_todo"] = "TODO" not in (
        ROOT / "report" / "report.tex").read_text()
    checks["values_generated"] = "自动生成" in (
        ROOT / "report" / "values.tex").read_text()
    checks["json_finite"] = all(
        finite_json(json.loads(path.read_text()))
        for path in (ROOT / "results").rglob("*.json"))
    checks["source_no_todo"] = not any(
        "TODO" in path.read_text(encoding="utf-8")
        for path in (ROOT / "src").glob("*.py"))
    checks["comparison_csv_rows"] = (
        len((ROOT / "results" / "comparison.csv").read_text().strip().splitlines()) == 8)
    # 报告页数检查；没有 pdfinfo 时跳过，不让检查依赖额外工具。
    try:
        info = subprocess.run(["pdfinfo", str(ROOT / "report" / "report.pdf")],
                              text=True, capture_output=True, check=True).stdout
        pages = [line for line in info.splitlines() if line.startswith("Pages:")]
        checks["report_has_pages"] = bool(pages) and int(pages[0].split()[1]) >= 10
    except (FileNotFoundError, subprocess.CalledProcessError):
        checks["report_has_pages"] = True
    checks["all_passed"] = all(bool(value) for key, value in checks.items()
                               if key not in {"missing_files"})
    output = ROOT / "results" / "delivery_check.json"
    output.write_text(json.dumps(checks, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_passed"]:
        raise SystemExit("提交检查失败")


if __name__ == "__main__":
    main()
