import torch

from src.classifier import (ClassifierModel, load_classifier_checkpoint,
                            parameter_report, save_classifier_checkpoint)
from src.model import DecoderLM, ModelConfig
from src.tokenizer import WordTokenizer


def tiny_setup():
    torch.manual_seed(0)
    tokenizer = WordTokenizer.build(["alpha beta gamma delta epsilon zeta"],
                                    min_freq=1)
    config = ModelConfig(vocab_size=len(tokenizer), context_length=24,
                         n_layers=2, d_model=16, n_heads=2, dropout=0.0)
    model = ClassifierModel(DecoderLM(config), num_classes=4, pooling="last",
                            head_dropout=0.0)
    return tokenizer, model


def test_classifier_shapes_and_loss():
    _, model = tiny_setup()
    input_ids = torch.randint(4, 10, (3, 7))
    lengths = torch.tensor([7, 5, 3])
    labels = torch.tensor([0, 2, 3])
    logits, loss = model(input_ids, lengths, labels)
    assert logits.shape == (3, 4)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model(input_ids, lengths)[1] is None


def test_last_pooling_ignores_right_padding():
    _, model = tiny_setup()
    model.eval()
    short = torch.randint(4, 10, (1, 4))
    padded = torch.cat([short, torch.randint(4, 10, (1, 5))], dim=1)
    short_logits, _ = model(short, torch.tensor([4]))
    padded_logits, _ = model(padded, torch.tensor([4]))
    torch.testing.assert_close(short_logits, padded_logits, rtol=1e-6, atol=1e-6)


def test_mean_pooling_masks_padding():
    _, model = tiny_setup()
    model.pooling = "mean"
    hidden = torch.randn(2, 6, 16)
    lengths = torch.tensor([6, 3])
    pooled = model.pool(hidden, lengths)
    expected = torch.stack([hidden[0].mean(0), hidden[1, :3].mean(0)])
    torch.testing.assert_close(pooled, expected)


def test_freeze_backbone_only_trains_head():
    _, model = tiny_setup()
    model.set_backbone_frozen(True)
    report = parameter_report(model)
    assert report["trainable"] == report["classification_head"] == 4 * 16 + 4
    assert report["frozen"] == report["backbone"]
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad
               for parameter in model.classification_head.parameters())
    model.train()
    assert model.classification_head.training
    assert not model.backbone.training
    input_ids = torch.randint(4, 10, (2, 5))
    lengths = torch.tensor([5, 3])
    labels = torch.tensor([1, 2])
    _, loss = model(input_ids, lengths, labels)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.classification_head.weight.grad is not None


def test_checkpoint_roundtrip(tmp_path):
    tokenizer, model = tiny_setup()
    model.set_backbone_frozen(False)
    path = tmp_path / "classifier.pt"
    save_classifier_checkpoint(path, model, tokenizer,
                               {"mode": "test", "epoch": 1})
    loaded, loaded_tokenizer, payload = load_classifier_checkpoint(path)
    assert payload["num_classes"] == 4
    assert loaded.pooling == "last"
    assert loaded_tokenizer.vocabulary == tokenizer.vocabulary
    model.eval()
    loaded.eval()
    input_ids = torch.randint(4, 10, (2, 6))
    torch.testing.assert_close(model(input_ids)[0], loaded(input_ids)[0])
