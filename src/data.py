"""AG News 数据集下载、解析、分层划分与词级编码。

数据来源（公开镜像，原始数据由 Zhang et al., 2015 发布）：
  https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

官方训练集 120,000 条、测试集 7,600 条，四类各占四分之一。本实验从官方训练集
中按类别分层抽取 10% 作为验证集，官方测试集只用于最终评估；词表、语言模型预训练
和分类器超参数选择都不接触测试集。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import urllib.request

import torch
from torch.utils.data import Dataset

from .tokenizer import (DEFAULT_LABEL_VERBALIZERS, TUNED_LABEL_VERBALIZERS,
                        WordTokenizer, basic_tokenize)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "ag_news_train.csv"
TEST_CSV = DATA_DIR / "ag_news_test.csv"
TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
VAL_FRACTION = 0.10
SEED = 42
MAX_LENGTH = 192
MIN_FREQ = 2
MAX_VOCAB = 30_000


@dataclass(frozen=True)
class Example:
    title: str
    description: str
    label: int

    @property
    def text(self) -> str:
        # 标题通常包含最直接的主题词，放在前面；空描述时不留多余空格。
        if self.description:
            return f"{self.title}. {self.description}"
        return self.title


@dataclass
class DataBundle:
    train: list[Example]
    validation: list[Example]
    test: list[Example]
    tokenizer: WordTokenizer
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_needed(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载 {url} -> {path}", flush=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(path)


def parse_csv(path: Path) -> list[Example]:
    """解析无表头的 AG News CSV：类别(1-4), 标题, 正文。"""
    examples = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"{path} 第 {line_number} 行字段数为 {len(row)}，应为 3")
            label = int(row[0]) - 1
            if not 0 <= label < len(LABELS):
                raise ValueError(f"{path} 第 {line_number} 行类别越界：{row[0]}")
            examples.append(Example(row[1].strip(), row[2].strip(), label))
    return examples


def stratified_split(examples: list[Example], validation_fraction: float,
                     seed: int) -> tuple[list[Example], list[Example]]:
    """按类别分层、使用固定种子打乱后切分，保证验证集类别比例稳定。"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 (0, 1) 内")
    rng = random.Random(seed)
    by_label = {label: [] for label in range(len(LABELS))}
    for example in examples:
        by_label[example.label].append(example)
    train, validation = [], []
    for label in range(len(LABELS)):
        group = list(by_label[label])
        rng.shuffle(group)
        n_validation = int(round(len(group) * validation_fraction))
        validation.extend(group[:n_validation])
        train.extend(group[n_validation:])
    # 再打乱一次，避免训练 batch 按类别聚集。
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def length_statistics(examples: list[Example], tokenizer: WordTokenizer,
                      max_length: int) -> dict:
    lengths = [len(basic_tokenize(example.text)) for example in examples]
    ordered = sorted(lengths)
    def percentile(q):
        # 最近秩法，避免插值引入不必要的小数噪声。
        index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
        return ordered[index]
    return {
        "count": len(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 2),
        "median_tokens": statistics.median(lengths),
        "p90_tokens": percentile(0.90),
        "p99_tokens": percentile(0.99),
        "max_tokens": max(lengths),
        "truncated_fraction": round(sum(length > max_length for length in lengths) / len(lengths), 6),
        "max_length": max_length,
    }


def class_distribution(examples: list[Example]) -> dict:
    counts = Counter(example.label for example in examples)
    return {LABELS[i]: counts.get(i, 0) for i in range(len(LABELS))}


def prepare_data(max_length: int = MAX_LENGTH, seed: int = SEED,
                 force_download: bool = False) -> DataBundle:
    """下载（若需要）、解析、划分并构建只在训练集上拟合的词表。"""
    if force_download:
        for path in (TRAIN_CSV, TEST_CSV):
            path.unlink(missing_ok=True)
    download_if_needed(TRAIN_CSV, TRAIN_URL)
    download_if_needed(TEST_CSV, TEST_URL)
    raw_train = parse_csv(TRAIN_CSV)
    test = parse_csv(TEST_CSV)
    if len(raw_train) != 120_000 or len(test) != 7_600:
        raise ValueError(f"数据规模异常：train={len(raw_train)}, test={len(test)}")
    train, validation = stratified_split(raw_train, VAL_FRACTION, seed)
    # 两套 verbalizer 都加入词表；实际使用哪套由验证集选择。
    extra_tokens = set(DEFAULT_LABEL_VERBALIZERS.values())
    extra_tokens.update(TUNED_LABEL_VERBALIZERS.values())
    extra_tokens.update({"topic", "title", "description"})
    # 词表只读取训练集文本，验证/测试集不参与。
    tokenizer = WordTokenizer.build(
        (example.text for example in train), min_freq=MIN_FREQ,
        max_vocab=MAX_VOCAB, extra_tokens=sorted(extra_tokens))
    metadata = {
        "dataset": "AG News",
        "source": "Zhang, Zhao & LeCun (2015), via CharCnn_Keras CSV mirror",
        "train_url": TRAIN_URL,
        "test_url": TEST_URL,
        "train_csv_sha256": _sha256(TRAIN_CSV),
        "test_csv_sha256": _sha256(TEST_CSV),
        "train_csv_bytes": TRAIN_CSV.stat().st_size,
        "test_csv_bytes": TEST_CSV.stat().st_size,
        "labels": LABELS,
        "num_classes": len(LABELS),
        "official_train_examples": len(raw_train),
        "validation_fraction": VAL_FRACTION,
        "split_seed": seed,
        "split": "official train stratified 90%/10% into train/validation; official test untouched",
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "class_distribution": {"train": class_distribution(train),
                               "validation": class_distribution(validation),
                               "test": class_distribution(test)},
        "text_construction": "title + '. ' + description",
        "tokenizer": {
            "type": "word-level, lowercased regex, training-only vocabulary",
            "min_freq": MIN_FREQ, "max_vocab": MAX_VOCAB,
            "vocab_size": len(tokenizer), "pad_id": tokenizer.pad_id,
            "unk_id": tokenizer.unk_id, "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id, "extra_tokens": sorted(extra_tokens),
        },
        "token_lengths": {
            "train": length_statistics(train, tokenizer, max_length),
            "validation": length_statistics(validation, tokenizer, max_length),
            "test": length_statistics(test, tokenizer, max_length),
        },
        "max_length_for_classification": max_length,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return DataBundle(train, validation, test, tokenizer, metadata)


class EncodedTextDataset(Dataset):
    """预分词后的分类数据集；右侧补 pad，真实长度单独保存。"""

    def __init__(self, examples: list[Example], tokenizer: WordTokenizer,
                 max_length: int):
        self.examples = examples
        self.texts = [example.text for example in examples]
        self.labels = torch.tensor([example.label for example in examples], dtype=torch.long)
        self.input_ids = []
        self.lengths = []
        for text in self.texts:
            ids = tokenizer.encode(text, max_length=max_length)
            if not ids:
                ids = [tokenizer.unk_id]
            self.input_ids.append(torch.tensor(ids, dtype=torch.long))
            self.lengths.append(len(ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.lengths[index]


def collate_classification(batch, pad_id: int):
    """把变长序列右侧补到 batch 内最大长度。

    因果注意力不会让最后一个有效词元看到右侧的 pad，因此“取最后一个有效词元”
    的池化结果不受补位长度影响；平均池化时再用 lengths 屏蔽 pad。
    """
    sequences, labels, lengths = zip(*batch)
    max_length = max(int(length) for length in lengths)
    padded = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, :len(sequence)] = sequence
    return (padded, torch.tensor(lengths, dtype=torch.long),
            torch.stack(labels))


def build_lm_stream(examples: list[Example], tokenizer: WordTokenizer,
                    max_document_tokens: int = 256) -> torch.Tensor:
    """把训练文档拼成一条词元流，文档之间用 <bos>/<eos> 分隔。

    语言模型预训练从这条流里随机截取连续窗口。文档之间加了边界词元，模型能学到
    “文档结束/开始”的统计规律；分类时不需要依赖整条流。
    """
    pieces = []
    for example in examples:
        ids = tokenizer.encode(example.text, max_length=max_document_tokens)
        pieces.extend([tokenizer.bos_id])
        pieces.extend(ids)
        pieces.extend([tokenizer.eos_id])
    return torch.tensor(pieces, dtype=torch.long)


def sample_lm_batch(stream: torch.Tensor, batch_size: int, context_length: int,
                    generator: torch.Generator, device):
    """采样 [B, T] 输入和右移一位的 [B, T] 目标。"""
    if len(stream) <= context_length:
        raise ValueError("训练流长度必须大于 context_length")
    starts = torch.randint(len(stream) - context_length, (batch_size,), generator=generator)
    offsets = torch.arange(context_length + 1)
    chunks = stream[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].contiguous().to(device), chunks[:, 1:].contiguous().to(device)


if __name__ == "__main__":
    bundle = prepare_data()
    print(json.dumps(bundle.metadata, ensure_ascii=False, indent=2))
