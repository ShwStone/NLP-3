"""训练脚本共用的小工具：随机种子、精度上下文、分类指标与 JSON 写出。"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import time
from zoneinfo import ZoneInfo

# 必须在创建 cuBLAS handle 前设置；否则确定性模式下的矩阵乘会直接报错。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 小模型 + 固定数据划分；确定性算法便于复现实验日志。
    torch.use_deterministic_algorithms(True)


def precision_context(device):
    """GPU 上使用 BF16 autocast，参数和优化器状态保持 FP32。"""
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext())


def now_beijing() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def environment_info(device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "deterministic_algorithms": True,
        "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "measured_at_beijing": now_beijing(),
    }


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def classification_metrics(y_true, y_pred, label_names) -> dict:
    """准确率、宏平均 F1、逐类指标和混淆矩阵。测试集评估统一走这里。"""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true 与 y_pred 必须等长且非空")
    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "num_examples": int(y_true.size),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> dict:
    """返回交叉熵损失（按样本加权）、预测和指标。"""
    model.eval()
    total_loss, total = 0.0, 0
    predictions, targets = [], []
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        with precision_context(device):
            logits, loss = model(input_ids, lengths, labels)
        total_loss += loss.item() * labels.numel()
        total += labels.numel()
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return {"loss": total_loss / total, "predictions": predictions,
            "targets": targets}


class Stopwatch:
    """只统计显式包裹的区间；CUDA 上先同步再计时。"""

    def __init__(self, device=None):
        self.device = device
        self.started = None
        self.elapsed = 0.0
        self.depth = 0

    def _synchronize(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        # 支持嵌套：外层统计总时间，内层评估不会重复累加。
        if self.depth == 0:
            self._synchronize()
            self.started = time.perf_counter()
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.depth -= 1
        if self.depth == 0:
            self._synchronize()
            self.elapsed += time.perf_counter() - self.started
            self.started = None
        return False

    def current(self, synchronize: bool = False) -> float:
        """返回从创建到现在的累计秒数；可用于在上下文内部写日志。"""
        if self.started is None:
            return self.elapsed
        if synchronize:
            self._synchronize()
        return self.elapsed + (time.perf_counter() - self.started)
