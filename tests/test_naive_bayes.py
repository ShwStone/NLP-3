import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.naive_bayes import (NBConfig, ScratchMultinomialNB, build_features)
from src.tokenizer import basic_tokenize


def toy_matrix():
    texts = ["bank market stock", "bank money market", "goal match team",
             "goal coach match", "stock market bank", "team goal coach"]
    labels = np.array([0, 0, 1, 1, 0, 1])
    vectorizer = CountVectorizer(tokenizer=basic_tokenize, lowercase=False,
                                 token_pattern=None)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, labels


def test_scratch_nb_matches_sklearn():
    _, matrix, labels = toy_matrix()
    scratch = ScratchMultinomialNB(alpha=0.5).fit(matrix, labels, 2)
    reference = MultinomialNB(alpha=0.5).fit(matrix, labels)
    np.testing.assert_allclose(scratch.class_log_prior,
                               reference.class_log_prior_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scratch.feature_log_prob,
                               reference.feature_log_prob_, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(scratch.predict(matrix), reference.predict(matrix))
    assert np.all(np.isfinite(scratch.predict_log_joint(matrix)))


def test_smoothing_handles_unseen_feature():
    _, matrix, labels = toy_matrix()
    model = ScratchMultinomialNB(alpha=1.0).fit(matrix, labels, 2)
    unseen = matrix.copy().tolil()
    unseen[:, 0] = 100.0
    log_joint = model.predict_log_joint(unseen.tocsr())
    assert np.isfinite(log_joint).all()


def test_features_fit_only_on_train_texts():
    train = ["alpha beta", "alpha gamma"]
    validation = ["alpha delta"]
    test = ["beta gamma"]
    vectorizer, train_matrix, validation_matrix, seconds = build_features(
        train, validation, NBConfig(alpha=0.1, ngram_max=1, min_df=1,
                                    max_features=50))
    test_matrix = vectorizer.transform(test)
    assert "delta" not in vectorizer.vocabulary_
    assert train_matrix.shape[0] == 2
    assert validation_matrix.shape[0] == 1
    assert test_matrix.shape[0] == 1
    assert seconds >= 0


def test_separable_toy_accuracy():
    _, matrix, labels = toy_matrix()
    model = ScratchMultinomialNB(alpha=0.1).fit(matrix, labels, 2)
    assert (model.predict(matrix) == labels).all()
