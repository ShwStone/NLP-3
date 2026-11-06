"""在作业二解码器主干上增加分类任务头。

两种池化方式：
  * ``last``：取每条样本最后一个真实词元的隐状态（右侧 pad 不会被因果注意力看到）；
  * ``mean``：对所有真实词元求平均，用 length mask 排除 pad。
分类头是单层线性映射，输出 ``num_classes`` 个 logits，训练目标为交叉熵。

``set_backbone_frozen(True)`` 只训练分类头，``False`` 则全参数微调。冻结模式下
``train()`` 会让主干保持 eval（关闭 dropout），避免线性探针学到带随机噪声的特征。
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import DecoderLM, ModelConfig
from .tokenizer import WordTokenizer


class ClassifierModel(nn.Module):
    def __init__(self, backbone: DecoderLM, num_classes: int,
                 pooling: str = "last", head_dropout: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 至少为 2")
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling 只能是 'last' 或 'mean'")
        self.backbone = backbone
        self.num_classes = num_classes
        self.pooling = pooling
        self.head_dropout = nn.Dropout(head_dropout)
        self.classification_head = nn.Linear(backbone.cfg.d_model, num_classes)
        # 小方差初始化，训练初期 logits 接近 0；偏置为 0，不偏向任何类别。
        nn.init.normal_(self.classification_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classification_head.bias)
        self.backbone_frozen = False

    def pool(self, hidden: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        batch, time, _ = hidden.shape
        if lengths is None:
            lengths = torch.full((batch,), time, dtype=torch.long, device=hidden.device)
        lengths = lengths.to(hidden.device).clamp(min=1, max=time)
        if self.pooling == "last":
            row = torch.arange(batch, device=hidden.device)
            return hidden[row, lengths - 1]
        # mean pooling：mask 形状 [B, T, 1]，分母是真实词元数。
        mask = torch.arange(time, device=hidden.device)[None, :] < lengths[:, None]
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths[:, None]

    def forward(self, input_ids, lengths=None, labels=None):
        hidden = self.backbone.encode(input_ids)
        pooled = self.pool(hidden, lengths)
        logits = self.classification_head(self.head_dropout(pooled))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return logits, loss

    def set_backbone_frozen(self, frozen: bool = True):
        self.backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        # 切换后立即同步 dropout 状态，避免调用方忘记设置。
        if frozen:
            self.backbone.eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结主干时保持 eval；分类头仍由父类设为 train。
        if self.backbone_frozen:
            self.backbone.eval()
        return self


def load_lm_checkpoint(path: str | Path, map_location="cpu"):
    """加载语言模型 checkpoint，返回 (checkpoint, tokenizer, backbone)。"""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    required = {"config", "state_dict", "tokenizer"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint 缺少字段：{sorted(missing)}")
    config = ModelConfig(**checkpoint["config"])
    tokenizer = WordTokenizer.from_state_dict(checkpoint["tokenizer"])
    if config.vocab_size != len(tokenizer):
        raise ValueError("checkpoint 的词表大小与 config 不一致")
    backbone = DecoderLM(config)
    backbone.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint, tokenizer, backbone


def build_classifier(config: ModelConfig, tokenizer: WordTokenizer,
                     num_classes: int, pooling: str = "last",
                     head_dropout: float = 0.1) -> ClassifierModel:
    backbone = DecoderLM(config)
    return ClassifierModel(backbone, num_classes, pooling, head_dropout)


def parameter_report(model: ClassifierModel) -> dict:
    """按对象 id 去重统计总参数、可训练参数和分类头参数。"""
    unique = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = id(parameter)
        if key in unique:
            unique[key]["aliases"].append(name)
            continue
        unique[key] = {"name": name, "shape": list(parameter.shape),
                       "count": parameter.numel(), "trainable": parameter.requires_grad,
                       "aliases": []}
    total = sum(entry["count"] for entry in unique.values())
    trainable = sum(entry["count"] for entry in unique.values() if entry["trainable"])
    head = sum(parameter.numel() for parameter in model.classification_head.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "backbone": total - head,
        "classification_head": head,
        "trainable_fraction": trainable / total,
        "tensors": len(unique),
    }


def save_classifier_checkpoint(path: str | Path, model: ClassifierModel,
                               tokenizer: WordTokenizer, extra: dict):
    payload = {
        "config": {
            "vocab_size": model.backbone.cfg.vocab_size,
            "context_length": model.backbone.cfg.context_length,
            "n_layers": model.backbone.cfg.n_layers,
            "d_model": model.backbone.cfg.d_model,
            "n_heads": model.backbone.cfg.n_heads,
            "dropout": model.backbone.cfg.dropout,
        },
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "num_classes": model.num_classes,
        "pooling": model.pooling,
        "backbone_frozen": model.backbone_frozen,
        "parameter_report": parameter_report(model),
        "extra": extra,
    }
    torch.save(payload, path)


def load_classifier_checkpoint(path: str | Path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**payload["config"])
    tokenizer = WordTokenizer.from_state_dict(payload["tokenizer"])
    backbone = DecoderLM(config)
    model = ClassifierModel(backbone, payload["num_classes"], payload["pooling"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.set_backbone_frozen(payload.get("backbone_frozen", False))
    return model, tokenizer, payload
