"""作业二解码器主干的语义测试；作业三的分类头复用它。"""
import torch
import pytest

from src.model import (CausalSelfAttention, DecoderLM, ManualLayerNorm,
                       ModelConfig, count_parameters, theoretical_parameters)
from src.data import sample_lm_batch


def config(**changes):
    values = {"vocab_size": 37, "context_length": 32, "n_layers": 2,
              "d_model": 32, "n_heads": 4, "dropout": 0.0}
    values.update(changes)
    return ModelConfig(**values)


def test_layernorm_values_and_gradients():
    torch.manual_seed(1)
    layer = ManualLayerNorm(8).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    target = torch.nn.functional.layer_norm(x, (8,), layer.scale, layer.shift, layer.eps)
    actual = layer(x)
    torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(x)
    grad_actual = torch.autograd.grad(
        (actual * weights).sum(), (x, layer.scale, layer.shift), retain_graph=True)
    grad_target = torch.autograd.grad((target * weights).sum(),
                                      (x, layer.scale, layer.shift))
    for left, right in zip(grad_actual, grad_target):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-9)
    assert torch.isfinite(layer(torch.ones_like(x))).all()


def test_attention_mask_and_manual_reference():
    torch.manual_seed(2)
    attention = CausalSelfAttention(config()).eval()
    x = torch.randn(2, 7, 32)
    actual, weights = attention(x, return_weights=True)
    assert (weights.triu(diagonal=1) == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 7))
    q, k, v = attention.qkv(x).chunk(3, dim=-1)
    q, k, v = [part.reshape(2, 7, 4, 8) for part in (q, k, v)]
    reference = torch.empty(2, 7, 4, 8)
    for batch in range(2):
        for head in range(4):
            for time in range(7):
                scores = (k[batch, :time + 1, head] * q[batch, time, head]).sum(-1) / (8 ** 0.5)
                reference[batch, time, head] = (scores.softmax(-1)[:, None]
                                                * v[batch, :time + 1, head]).sum(0)
    torch.testing.assert_close(actual, attention.output(reference.reshape(2, 7, 32)))


def test_future_token_invariance_and_encode_consistency():
    torch.manual_seed(3)
    model = DecoderLM(config()).eval()
    ids = torch.randint(37, (2, 24))
    modified = ids.clone()
    modified[:, 12:] = (modified[:, 12:] + 1) % 37
    first = model(ids)[0]
    second = model(modified)[0]
    torch.testing.assert_close(first[:, :12], second[:, :12], rtol=0, atol=0)
    torch.testing.assert_close(first[:, :12], model(ids[:, :12])[0],
                               rtol=1e-5, atol=1e-6)
    # encode 必须与 forward 使用完全相同的主干表示。
    torch.testing.assert_close(model.lm_head(model.encode(ids)), first)


def test_parameter_count_deduplicates_tied_head():
    model = DecoderLM(config())
    count = count_parameters(model)
    assert model.lm_head.weight is model.token_embedding.weight
    assert count["total"] == theoretical_parameters(model.cfg)
    assert count["total"] == sum(parameter.numel() for parameter in model.parameters())
    embedding_entry = next(entry for entry in count["tensors"]
                           if entry["name"] == "token_embedding.weight")
    assert "lm_head.weight" in embedding_entry["aliases"]


def test_shifted_lm_batch_and_generation_crop():
    stream = torch.arange(300)
    x, y = sample_lm_batch(stream, 4, 16, torch.Generator().manual_seed(0), "cpu")
    assert torch.equal(y, x + 1)
    model = DecoderLM(config(context_length=8)).train()
    prompt = torch.zeros(1, 12, dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=4,
                               generator=torch.Generator().manual_seed(5))
    assert generated.shape == (1, 16)
    assert torch.equal(generated[:, :12], prompt)
    with pytest.raises(ValueError):
        model(prompt)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, d_model=33, n_heads=4)
