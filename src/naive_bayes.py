"""多项式朴素贝叶斯文本分类基线。

流程：
  1. 用与神经模型相同的正则分词器切词（小写、标点单独成词）；
  2. 在训练集上统计 unigram + bigram 词袋，过滤低频特征；
  3. 用拉普拉斯平滑估计每个类别下的词概率，取对数后做线性判别；
  4. 在验证集上选择平滑系数 alpha，最后只在测试集评估一次。

这里故意把 NB 的加平滑、取对数、先验和预测写成显式实现，而不是直接调用
``MultinomialNB``；同时用 sklearn 的实现做数值交叉核对。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .data import LABELS, ROOT, prepare_data
from .tokenizer import basic_tokenize
from .utils import Stopwatch, classification_metrics, environment_info, write_json


@dataclass
class NBConfig:
    alpha: float = 1.0
    ngram_max: int = 2
    min_df: int = 3
    max_features: int = 200_000


class ScratchMultinomialNB:
    """显式多项式 NB：log P(c) 与 log P(w|c) 都是可解释的数组。"""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha 必须为正")
        self.alpha = alpha
        self.class_log_prior = None
        self.feature_log_prob = None
        self.class_count = None
        self.feature_count = None

    def fit(self, X, y, n_classes: int):
        X = sparse.csr_matrix(X)
        y = np.asarray(y, dtype=np.int64)
        if X.shape[0] != y.size:
            raise ValueError("特征行数与标签数不一致")
        n_features = X.shape[1]
        self.class_count = np.bincount(y, minlength=n_classes).astype(np.int64)
        if np.any(self.class_count == 0):
            raise ValueError("每个类别至少需要一个训练样本")
        self.class_log_prior = np.log(self.class_count / y.size)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        for label in range(n_classes):
            # 稀疏矩阵按行求和得到该类别的词频向量。
            feature_count[label] = np.asarray(
                X[y == label].sum(axis=0), dtype=np.float64).ravel()
        self.feature_count = feature_count
        smoothed = feature_count + self.alpha
        self.feature_log_prob = np.log(
            smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_log_joint(self, X):
        X = sparse.csr_matrix(X)
        if self.feature_log_prob is None:
            raise RuntimeError("请先调用 fit")
        return np.asarray(X @ self.feature_log_prob.T) + self.class_log_prior

    def predict(self, X):
        return self.predict_log_joint(X).argmax(axis=1)

    def predict_proba(self, X):
        log_joint = self.predict_log_joint(X)
        # 数值稳定的 softmax。
        shifted = log_joint - log_joint.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def build_features(train_texts, validation_texts, config: NBConfig):
    """在训练集拟合词袋，验证集只 transform；测试集在推理计时内 transform。"""
    vectorizer = CountVectorizer(
        tokenizer=basic_tokenize, preprocessor=None, lowercase=False,
        token_pattern=None, ngram_range=(1, config.ngram_max),
        min_df=config.min_df, max_features=config.max_features,
        dtype=np.float32)
    with Stopwatch() as timer:
        train_matrix = vectorizer.fit_transform(train_texts)
        validation_matrix = vectorizer.transform(validation_texts)
    return vectorizer, train_matrix, validation_matrix, timer.elapsed


def top_features(vectorizer, model: ScratchMultinomialNB, limit: int = 12):
    """用“该类对数概率 - 其余类平均对数概率”展示最有判别力的 n-gram。"""
    names = vectorizer.get_feature_names_out()
    output = {}
    for label, name in enumerate(LABELS):
        scores = model.feature_log_prob[label] - (
            model.feature_log_prob.sum(axis=0) - model.feature_log_prob[label]
        ) / (len(LABELS) - 1)
        best = np.argsort(scores)[::-1][:limit]
        output[name] = [{"feature": str(names[i]), "score": round(float(scores[i]), 5)}
                        for i in best]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "results" / "naive_bayes"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = prepare_data()
    train_texts = [example.text for example in bundle.train]
    validation_texts = [example.text for example in bundle.validation]
    test_texts = [example.text for example in bundle.test]
    train_labels = np.array([example.label for example in bundle.train])
    validation_labels = np.array([example.label for example in bundle.validation])
    test_labels = np.array([example.label for example in bundle.test])
    base = NBConfig(ngram_max=args.ngram_max, min_df=args.min_df,
                    max_features=args.max_features)
    vectorizer, train_matrix, validation_matrix, feature_seconds = \
        build_features(train_texts, validation_texts, base)
    feature_names = vectorizer.get_feature_names_out()
    print(f"词袋特征数：{len(feature_names)}；训练矩阵 nnz={train_matrix.nnz}", flush=True)

    # 只用验证集选择 alpha，测试集不参与。
    candidates = [0.1, 0.3, 1.0, 3.0]
    tuning = []
    best_alpha, best_val_accuracy = None, -1.0
    for alpha in candidates:
        model = ScratchMultinomialNB(alpha=alpha).fit(
            train_matrix, train_labels, len(LABELS))
        predictions = model.predict(validation_matrix)
        accuracy = float((predictions == validation_labels).mean())
        tuning.append({"alpha": alpha, "validation_accuracy": accuracy})
        print(f"alpha={alpha:<4} val_acc={accuracy:.5f}", flush=True)
        if accuracy > best_val_accuracy:
            best_alpha, best_val_accuracy = alpha, accuracy
    config = NBConfig(alpha=best_alpha, ngram_max=args.ngram_max,
                      min_df=args.min_df, max_features=args.max_features)

    # 用选定的 alpha 重新在训练集上拟合；计时只算这一步，特征构建另计。
    with Stopwatch() as fit_timer:
        model = ScratchMultinomialNB(alpha=config.alpha).fit(
            train_matrix, train_labels, len(LABELS))
    with Stopwatch() as inference_timer:
        # NB 的推理包含词袋 transform 和线性判别，二者都计入测试耗时。
        test_matrix = vectorizer.transform(test_texts)
        test_predictions = model.predict(test_matrix)
    test_metrics = classification_metrics(test_labels, test_predictions, LABELS)

    # sklearn 交叉核对：同一特征矩阵、同一 alpha，预测应几乎完全一致。
    sklearn_model = MultinomialNB(alpha=config.alpha).fit(train_matrix, train_labels)
    sklearn_predictions = sklearn_model.predict(test_matrix)
    sklearn_accuracy = float((sklearn_predictions == test_labels).mean())
    agreement = float((sklearn_predictions == test_predictions).mean())
    validation_predictions = model.predict(validation_matrix)
    validation_metrics = classification_metrics(
        validation_labels, validation_predictions, LABELS)

    n_features = train_matrix.shape[1]
    parameter_count = int(n_features * len(LABELS) + len(LABELS))
    metrics = {
        "model": "Multinomial Naive Bayes",
        "feature_config": {
            "tokenizer": "same lowercased regex as WordTokenizer",
            "ngram_range": [1, config.ngram_max],
            "min_df": config.min_df, "max_features": config.max_features,
            "n_features": int(n_features),
        },
        "alpha": config.alpha,
        "alpha_candidates": candidates,
        "validation_tuning": tuning,
        "validation": validation_metrics,
        "test": test_metrics,
        "parameters": {
            "learned_log_probabilities": int(n_features * len(LABELS)),
            "class_log_priors": len(LABELS),
            "total": parameter_count,
            "note": "NB 的可学习量是每个类别每个特征的对数概率，不是神经网络权重",
        },
        "feature_matrix": {
            "train_rows": int(train_matrix.shape[0]),
            "train_nnz": int(train_matrix.nnz),
            "train_density": float(train_matrix.nnz / (train_matrix.shape[0] * n_features)),
            "test_rows": int(test_matrix.shape[0]),
            "test_nnz": int(test_matrix.nnz),
        },
        "train_seconds": {
            "feature_extraction_and_transform": feature_seconds,
            "nb_fit": fit_timer.elapsed,
            "total": feature_seconds + fit_timer.elapsed,
        },
        "inference_seconds_including_transform": inference_timer.elapsed,
        "test_examples_per_second": len(test_texts) / inference_timer.elapsed,
        "sklearn_crosscheck": {
            "accuracy": sklearn_accuracy,
            "prediction_agreement": agreement,
            "accuracy_difference": abs(sklearn_accuracy - test_metrics["accuracy"]),
        },
        "top_features": top_features(vectorizer, model),
        "environment": environment_info(__import__("torch").device("cpu")),
        "data": {
            "counts": bundle.metadata["counts"],
            "train_csv_sha256": bundle.metadata["train_csv_sha256"],
            "test_csv_sha256": bundle.metadata["test_csv_sha256"],
        },
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "test_predictions.json", {
        "targets": test_labels.tolist(),
        "predictions": test_predictions.tolist(),
        "probabilities": model.predict_proba(test_matrix).tolist(),
    })
    # 保存特征名和每类 top 特征，便于报告解释。
    write_json(output / "top_features.json", metrics["top_features"])
    print(json.dumps({
        "alpha": config.alpha, "n_features": int(n_features),
        "parameters": parameter_count,
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_seconds": metrics["train_seconds"],
        "inference_seconds": metrics["inference_seconds_including_transform"],
        "sklearn_accuracy": sklearn_accuracy,
        "prediction_agreement": agreement,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
