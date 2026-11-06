"""作业三使用的词级分词器。

作业二采用字符级分词；分类任务中 AG News 的一条新闻平均只有几十个词，若继续按
字符切分，192 个位置只能覆盖约一半文本。这里把作业二的“训练集构建词表”逻辑
扩展到词级：小写化、正则分词、按训练集词频保留高频词，未见词映射为 <unk>。
分词器完全不使用验证集或测试集，避免词表泄漏。

提示学习需要用类别词作为“口头化标签”（verbalizer）。这些词由调用方显式加入
词表，保证它们不会被 <unk> 替代；它们本身不是监督信号，AG News 的文本里也
经常自然出现 world/sports/business/science。
"""
from __future__ import annotations

from collections import Counter
import re

# 单词（允许内部撇号）或单个非空白符号。先小写再切分，标点单独成词。
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]", re.IGNORECASE)


def basic_tokenize(text: str) -> list[str]:
    """返回小写词元列表；不依赖 NLTK/spaCy 等外部分词包。"""
    return _TOKEN_PATTERN.findall(text.lower())


class WordTokenizer:
    """固定词表的可逆（对词表内词元）词级分词器。"""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIALS = (PAD, UNK, BOS, EOS)

    def __init__(self, vocabulary: list[str]):
        # 保留调用方给出的顺序；特殊词元必须位于最前面，便于固定 id。
        if tuple(vocabulary[:4]) != self.SPECIALS:
            raise ValueError("词表前四项必须是 <pad>, <unk>, <bos>, <eos>")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("词表含重复词元")
        self.vocabulary = list(vocabulary)
        self.stoi = {token: i for i, token in enumerate(self.vocabulary)}
        self.itos = list(self.vocabulary)
        self.pad_id = self.stoi[self.PAD]
        self.unk_id = self.stoi[self.UNK]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]

    @classmethod
    def build(cls, texts, min_freq: int = 2, max_vocab: int = 30_000,
              extra_tokens=()):
        """仅从给定文本（训练集）统计词频并构建词表。

        ``extra_tokens`` 用于加入任务模板词和类别 verbalizer；它们被放在高频词
        之后、低频词之前，不改变训练文本的统计过程。
        """
        if min_freq < 1 or max_vocab < len(cls.SPECIALS):
            raise ValueError("min_freq 必须为正，max_vocab 不能小于特殊词元数")
        counts = Counter()
        for text in texts:
            counts.update(basic_tokenize(text))
        # 先按频次降序、再按字典序，保证不同运行之间完全确定。
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        vocabulary = list(cls.SPECIALS)
        seen = set(vocabulary)
        for token, frequency in ordered:
            if frequency < min_freq or len(vocabulary) >= max_vocab:
                break
            if token not in seen:
                vocabulary.append(token)
                seen.add(token)
        for token in extra_tokens:
            token = str(token).lower()
            if token not in seen and len(vocabulary) < max_vocab:
                vocabulary.append(token)
                seen.add(token)
        return cls(vocabulary)

    def encode(self, text: str, max_length: int | None = None,
               add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """编码文本；长度限制作用于正文，之后再加 BOS/EOS。"""
        ids = [self.stoi.get(token, self.unk_id) for token in basic_tokenize(text)]
        if max_length is not None:
            if max_length < 1:
                raise ValueError("max_length 必须为正")
            # 长度限制针对最终序列；截断正文时保留开头，并给特殊词元留位置。
            reserved = int(add_bos) + int(add_eos)
            if max_length < reserved:
                raise ValueError("max_length 不足以容纳 BOS/EOS")
            ids = ids[:max_length - reserved]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        words = []
        for index in ids:
            token = self.itos[int(index)]
            # 跳过结构性特殊词元，但保留 <unk>，便于发现词表外输入。
            if skip_special and token in (self.PAD, self.BOS, self.EOS):
                continue
            words.append(token)
        return " ".join(words)

    def state_dict(self):
        return {"vocabulary": list(self.vocabulary), "type": "word-level-v1"}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["vocabulary"])

    def __len__(self):
        return len(self.vocabulary)


# 默认 verbalizer 使用最直接的类别词；调优版本由验证集上的提示词选择实验得到，
# 不使用测试集。两套词都加入词表，保证切换 preset 时不会出现 <unk>。
DEFAULT_LABEL_VERBALIZERS = {
    "World": "world",
    "Sports": "sports",
    "Business": "business",
    "Sci/Tech": "science",
}
TUNED_LABEL_VERBALIZERS = {
    "World": "politics",
    "Sports": "sports",
    "Business": "financial",
    "Sci/Tech": "computers",
}


def label_verbalizers(preset: str = "tuned"):
    """返回类别到口头化标签词的映射；preset 为 default 或 tuned。"""
    if preset == "default":
        return dict(DEFAULT_LABEL_VERBALIZERS)
    if preset == "tuned":
        return dict(TUNED_LABEL_VERBALIZERS)
    raise ValueError("preset 只能是 'default' 或 'tuned'")
