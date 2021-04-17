#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
from mylearn.tree_model import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder


class GradientBoost(BaseEstimator):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def preprocess_y(self, y):
        raise NotImplementedError

    def negative_gradient(self, predict, y):
        raise NotImplementedError

    def postprocess_predictions(self, predictions):
        raise NotImplementedError

    def fit(self, X, y):
        # 对于分类器，可能需要做 OHE
        y = self.preprocess_y(y)
        n_classes = y.shape[1]
        self.n_classes = n_classes
        # GBDT 是一个加法模型，预测值为所有的模型预测值之和
        self.estimators = []
        predictions = np.zeros([X.shape[0], n_classes])
        for i in range(self.n_estimators):
            cur_estimators = []
            target = self.negative_gradient(predictions, y)
            if target.ndim == 1:
                target = target[:, np.newaxis]
            for k in range(n_classes):
                tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, target[:, k])
                cur_estimators.append(tree)
                # 更新预测值
                predictions[:, k] += self.learning_rate * tree.predict(X)
            self.estimators.append(cur_estimators)
        return self

    def _predict(self, X):
        n_classes = self.n_classes
        predictions = np.zeros([X.shape[0], n_classes])
        for i in range(self.n_estimators):
            cur_estimators = self.estimators[i]
            for k, tree in enumerate(cur_estimators):
                predictions[:, k] += self.learning_rate * tree.predict(X)
        return predictions

    def predict(self, X):
        return self.postprocess_predictions(self._predict(X))


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class GradientBoostRegressor(GradientBoost, RegressorMixin):
    def preprocess_y(self, y):
        return y[:, np.newaxis]

    def negative_gradient(self, predict, y):
        return y - predict

    def postprocess_predictions(self, predictions):
        return predictions.flatten()


class GradientBoostClassifier(GradientBoost, ClassifierMixin):
    def preprocess_y(self, y):
        return OneHotEncoder(sparse=False, dtype="int32").fit_transform(y[:, np.newaxis])

    def negative_gradient(self, predict, y):
        return y - sigmoid(predict)

    def postprocess_predictions(self, predictions):
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        predictions = self._predict(X)
        return predictions / np.sum(predictions, axis=1)
