import torch

from src.data import (EncodedTextDataset, Example, collate_classification,
                      length_statistics, stratified_split)
from src.tokenizer import WordTokenizer


def synthetic_examples():
    examples = []
    for label in range(4):
        for index in range(20):
            examples.append(Example(f"title {label} {index}",
                                    f"description word{label} item{index}", label))
    return examples


def test_stratified_split_is_disjoint_and_balanced():
    examples = synthetic_examples()
    train, validation = stratified_split(examples, 0.25, seed=7)
    assert len(train) == 60 and len(validation) == 20
    assert len({id(example) for example in train + validation}) == 80
    for label in range(4):
        assert sum(example.label == label for example in validation) == 5
        assert sum(example.label == label for example in train) == 15


def test_length_statistics_reports_truncation():
    tokenizer = WordTokenizer.build(["a b c d e"], min_freq=1)
    examples = [Example("a b c", "", 0), Example("a b c d e", "", 1)]
    stats = length_statistics(examples, tokenizer, max_length=4)
    assert stats["count"] == 2
    assert stats["max_tokens"] == 5
    assert stats["truncated_fraction"] == 0.5


def test_encoded_dataset_and_collate_padding():
    tokenizer = WordTokenizer.build(["alpha beta gamma"], min_freq=1)
    examples = [Example("alpha beta", "", 0), Example("alpha", "", 1)]
    dataset = EncodedTextDataset(examples, tokenizer, max_length=8)
    input_ids, lengths, labels = collate_classification(
        [dataset[0], dataset[1]], tokenizer.pad_id)
    assert input_ids.shape == (2, 2)
    assert torch.equal(labels, torch.tensor([0, 1]))
    assert torch.equal(lengths, torch.tensor([2, 1]))
    assert input_ids[1, 1] == tokenizer.pad_id
    # 最大长度截断：3 个词截到 2 个。
    dataset = EncodedTextDataset([Example("alpha beta gamma", "", 0)],
                                 tokenizer, max_length=2)
    assert dataset.lengths == [2]
