"""生成提交压缩包和 SHA-256 清单。

包含完整源码、测试、脚本、报告、指标/日志/样例/核验结果；不包含
data/*.csv（脚本可下载）和 results/**/*.pt（体积较大，可由实验重建）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "编程作业三_完整提交包.zip"


def collect_files():
    files = []
    for directory in ("src", "tests", "scripts"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    for name in ("README.md", "requirements.txt", "pytest.ini", "实验报告_文本分类.pdf"):
        path = ROOT / name
        if path.exists():
            files.append(path)
    files.extend(path for path in (ROOT / "data").glob("metadata.json"))
    for path in (ROOT / "results").rglob("*"):
        if path.is_file() and path.suffix != ".pt":
            files.append(path)
    report = ROOT / "report"
    for name in ("report.tex", "values.tex", "report.pdf"):
        path = report / name
        if path.exists():
            files.append(path)
    for suffix in ("rows.tex",):
        files.extend(path for path in report.glob(f"*{suffix}"))
    files.extend(path for path in (report / "figures").glob("*.pdf"))
    files.extend(path for path in (report / "figures").glob("*.png"))
    # 去重并只保留项目内文件。
    unique = {}
    for path in files:
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        if str(relative) == "results/package_manifest.json":
            continue
        unique[str(relative)] = path
    return sorted(unique.items())


def main():
    if not (ROOT / "report" / "report.pdf").exists():
        raise SystemExit("缺少 report/report.pdf，请先运行 bash scripts/build_report.sh")
    files = collect_files()
    manifest_lines = []
    for relative, path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {relative}")
    manifest = "\n".join(manifest_lines) + "\n"
    (ROOT / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for relative, path in files:
            archive.write(path, arcname=relative)
        archive.writestr("MANIFEST.sha256", manifest)
    package = {
        "archive": ARCHIVE.name,
        "size_bytes": ARCHIVE.stat().st_size,
        "size_mib": ARCHIVE.stat().st_size / 2 ** 20,
        "file_count": len(files) + 1,
        "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "excluded": ["data/*.csv", "results/**/*.pt"],
    }
    (ROOT / "results" / "package_manifest.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
