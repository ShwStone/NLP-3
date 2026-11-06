import torch

from src.data import Example
from src.model import DecoderLM, ModelConfig
from src.prompt import (calibration_scores, make_sequences, score_batch,
                        select_exemplars)
from src.tokenizer import WordTokenizer


def setup():
    torch.manual_seed(0)
    tokenizer = WordTokenizer.build(
        ["title description topic world sports business science alpha beta"],
        min_freq=1)
    config = ModelConfig(vocab_size=len(tokenizer), context_length=40,
                         n_layers=2, d_model=16, n_heads=2, dropout=0.0)
    model = DecoderLM(config).eval()
    labels = [tokenizer.encode(word) for word in
              ("world", "sports", "business", "science")]
    return tokenizer, model, labels


def test_sequence_construction_and_budget():
    tokenizer, _, labels = setup()
    examples = [Example("alpha beta", "alpha beta beta", 0),
                Example("beta", "alpha", 2)]
    reserve = max(len(ids) for ids in labels) + 1
    sequences, positions, owners = make_sequences(
        examples, tokenizer, labels, [], 40, reserve)
    assert len(sequences) == 8
    assert len(positions) == 8
    assert owners == [0, 1, 2, 3] * 2
    for sequence in sequences:
        assert len(sequence) <= 40
    # 每个标签位置的前一个位置是实际用于预测的 logits 位置。
    for sequence, target_positions in zip(sequences, positions):
        assert target_positions[-1] == len(sequence) - 1
        assert sequence[target_positions[0]:] == sequence[-2:]


def test_score_batch_matches_manual_log_probability():
    tokenizer, model, labels = setup()
    examples = [Example("alpha beta", "alpha beta", 0)]
    reserve = max(len(ids) for ids in labels) + 1
    sequences, positions, owners = make_sequences(
        examples, tokenizer, labels, [], 40, reserve)
    scores = score_batch(model, tokenizer, sequences, positions, owners, 4,
                         torch.device("cpu"), 40)
    assert scores.shape == (1, 4)
    sequence = sequences[0]
    ids = torch.tensor([sequence])
    hidden = model.encode(ids)
    log_probs = torch.log_softmax(model.lm_head(hidden).float(), dim=-1)
    manual = []
    for position in positions[0]:
        manual.append(log_probs[0, position - 1, sequence[position]].item())
    torch.testing.assert_close(scores[0, 0], torch.tensor(sum(manual) / len(manual)),
                               rtol=1e-5, atol=1e-6)


def test_calibration_shape_and_exemplars():
    tokenizer, model, labels = setup()
    calibration = calibration_scores(model, tokenizer, labels, [], 40, 2,
                                     torch.device("cpu"))
    assert calibration.shape == (4,)
    training = [
        Example("alpha beta", "alpha beta beta", 0),
        Example("beta alpha", "beta alpha alpha", 1),
        Example("alpha alpha", "beta beta alpha", 2),
        Example("beta beta", "alpha alpha beta", 3),
    ]
    exemplars = select_exemplars(training, tokenizer, per_class=1,
                                 max_tokens=12, verbalizer_preset="tuned")
    assert all(len(exemplars[label]) == 1 for label in
               ("World", "Sports", "Business", "Sci/Tech"))
