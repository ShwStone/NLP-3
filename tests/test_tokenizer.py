import pytest

from src.tokenizer import (DEFAULT_LABEL_VERBALIZERS, TUNED_LABEL_VERBALIZERS,
                           WordTokenizer, basic_tokenize, label_verbalizers)


def test_basic_tokenize_lowercase_and_punctuation():
    assert basic_tokenize("Hello, WORLD! It's 2026.") == [
        "hello", ",", "world", "!", "it's", "2026", "."]
    assert basic_tokenize("U.S. stocks") == ["u", ".", "s", ".", "stocks"]


def test_build_vocab_only_uses_given_texts_and_extras():
    tokenizer = WordTokenizer.build(
        ["alpha beta beta", "beta gamma"], min_freq=2,
        extra_tokens=["politics", "computers"])
    assert tokenizer.vocabulary[:4] == list(WordTokenizer.SPECIALS)
    assert "beta" in tokenizer.stoi
    assert "alpha" not in tokenizer.stoi  # 只出现一次，被 min_freq 过滤
    assert tokenizer.encode("alpha")[0] == tokenizer.unk_id
    assert tokenizer.stoi["politics"] > 0
    assert tokenizer.stoi["computers"] > 0
    assert len(set(tokenizer.vocabulary)) == len(tokenizer.vocabulary)


def test_encode_truncation_and_special_tokens():
    tokenizer = WordTokenizer.build(["a b c d e f z"], min_freq=1)
    assert tokenizer.encode("a b c") == [tokenizer.stoi[word] for word in ("a", "b", "c")]
    ids = tokenizer.encode("a b c d e", max_length=5, add_bos=True, add_eos=True)
    assert len(ids) == 5
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids, skip_special=True) == "a b c"
    with pytest.raises(ValueError):
        tokenizer.encode("a", max_length=1, add_bos=True, add_eos=True)
    assert tokenizer.decode(tokenizer.encode("a unknownword z")) == "a <unk> z"


def test_verbalizer_presets_are_single_known_tokens():
    tokenizer = WordTokenizer.build(
        ["news text"], min_freq=1,
        extra_tokens=set(DEFAULT_LABEL_VERBALIZERS.values())
        | set(TUNED_LABEL_VERBALIZERS.values()))
    for preset in ("default", "tuned"):
        mapping = label_verbalizers(preset)
        assert set(mapping) == {"World", "Sports", "Business", "Sci/Tech"}
        for word in mapping.values():
            ids = tokenizer.encode(word)
            assert len(ids) == 1
            assert ids[0] != tokenizer.unk_id
    with pytest.raises(ValueError):
        label_verbalizers("unknown")
