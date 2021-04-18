#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter
import numpy as np
from heapq import heappush, heappop
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder


def _gain(g, h, lambda_l2):
    # 分子 g^2
    nominator = np.square(g.sum())
    # 分母 h+λ
    denominator = h.sum() + lambda_l2
    return 0.5 * (nominator / denominator)


class XgbTreeNode:
    def __init__(self, X, g, h, lambda_l2, gamma, depth):
        self.gamma = gamma
        self.lambda_l2 = lambda_l2
        self.h = h
        self.g = g
        self.depth = depth
        self.X = X
        self.left_node = None
        self.right_node = None
        self.is_leaf = True
        # 直接算出当前叶子结点的值（相比于决策树是用y算的，这里是用g h 算的）
        self.leaf_value = -g.sum() / (h.sum() + self.lambda_l2)

    def find_best_split_point(self):
        '''
        1. 寻找最优分裂点
        2. 并计算当前结点的信息增益值
        '''
        X = self.X
        g = self.g
        h = self.h
        lambda_l2 = self.lambda_l2
        gamma = self.gamma
        n_features = X.shape[1]
        best_feature_ix = -1
        best_threshold = -1
        best_gain = -np.inf
        # 当前的信息增益（损失值取负）
        cur_gain = _gain(g, h, self.lambda_l2)
        # 对每个特征进行遍历
        for feature_ix in range(n_features):
            feature = X[:, feature_ix]
            # 排序后的特征
            sorted_feature = np.unique(feature)
            # 特征只有一个取值，不管
            if sorted_feature.size == 1:
                continue
            # 根据连续特征的分为点得到【thresholds】
            thresholds = (sorted_feature[:1] + sorted_feature[1:]) / 2
            for threshold in thresholds:
                g_left = g[feature <= threshold]
                g_right = g[feature > threshold]
                h_left = h[feature <= threshold]
                h_right = h[feature > threshold]
                left_gain = _gain(g_left, h_left, lambda_l2)
                right_gain = _gain(g_right, h_right, lambda_l2)
                gain = left_gain + right_gain - cur_gain - gamma
                if gain > best_gain:
                    best_feature_ix = feature_ix
                    best_gain = gain
                    best_threshold = threshold
        self.gain = best_gain
        self.feature_ix = best_feature_ix
        self.threshold = best_threshold

    def split(self):
        '''
        结点分裂
        :return: 返回两个叶子结点
        '''
        left_mask = (self.X[:, self.feature_ix] <= self.threshold)
        right_mask = (self.X[:, self.feature_ix] > self.threshold)
        left_node = XgbTreeNode(
            self.X[left_mask, :], self.g[left_mask], self.h[left_mask],
            self.lambda_l2, self.gamma,
            self.depth + 1)
        left_node.find_best_split_point()
        right_node = XgbTreeNode(
            self.X[right_mask, :], self.g[right_mask], self.h[right_mask],
            self.lambda_l2, self.gamma,
            self.depth + 1)
        right_node.find_best_split_point()
        self.left_node = left_node
        self.right_node = right_node
        self.is_leaf = False
        return left_node, right_node

    def predict(self, X):
        if self.is_leaf:
            return self.leaf_value
        if X[self.feature_ix] <= self.threshold:
            return self.left_node.predict(X)
        else:
            return self.right_node.predict(X)

    def __lt__(self, other):
        '''用于结点之间的比较，信息增益(gain)大的，值小'''
        return self.gain > other.gain


class XgbDecisionTree(BaseEstimator):
    def __init__(self, max_depth=5, lambda_l2=0.1, gamma=1):
        self.lambda_l2 = lambda_l2
        self.gamma = gamma
        self.max_depth = max_depth

    def fit(self, X, g, h):
        '''
        根据数据、一阶导、二阶导 拟合一棵树（不是直接拟合）
        :param X: 数据（n_samples x n_features）
        :param g: gradient 梯度
        :param h: hessian  黑塞矩阵，二阶导
        :return:
        '''
        self.root = XgbTreeNode(X, g, h, self.lambda_l2, self.gamma, 0)
        self.root.find_best_split_point()
        # 用一个优先队列维护树的叶子结点，每次选取信息增益最大的结点做下一次的候选点
        heap = []
        heappush(heap, self.root)  # 大根堆
        while heap:
            node = heappop(heap)
            if node.depth >= self.max_depth:
                continue
            if np.isinf(node.gain):
                break
            left, right = node.split()
            heappush(heap, left)
            heappush(heap, right)
        return self

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.root.predict(X[i]))
        return np.array(predictions)


class XgbModel(BaseEstimator):
    def __init__(self, n_estimators=10, learning_rate=0.1,
                 max_depth=5, lambda_l2=0.1, gamma=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.lambda_l2 = lambda_l2
        self.max_depth = max_depth

    def preprocess_y(self, y):
        raise NotImplementedError

    def gradient(self, predict, y):
        raise NotImplementedError

    def hessian(self, predict, y):
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
            g = self.gradient(predictions, y)
            h = self.hessian(predictions, y)
            if g.ndim == 1:
                g = g[:, np.newaxis]
                h = h[:, np.newaxis]
            for k in range(n_classes):
                tree = XgbDecisionTree(
                    max_depth=self.max_depth, lambda_l2=self.lambda_l2,
                    gamma=self.gamma
                ).fit(X, g[:, k], h[:, k])
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


class XgbRegressor(XgbModel, RegressorMixin):
    def preprocess_y(self, y):
        return y[:, np.newaxis]

    def gradient(self, predict, y):
        return predict - y

    def hessian(self, predict, y):
        return np.ones_like(y)

    def postprocess_predictions(self, predictions):
        return predictions.flatten()


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class XgbClassifier(XgbModel, ClassifierMixin):
    def preprocess_y(self, y):
        return OneHotEncoder(sparse=False, dtype="int32").fit_transform(y[:, np.newaxis])

    def gradient(self, predict, y):
        return sigmoid(predict) - y

    def hessian(self, predict, y):
        return sigmoid(predict)*(1 - sigmoid(predict))

    def postprocess_predictions(self, predictions):
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        predictions = self._predict(X)
        return predictions / np.sum(predictions, axis=1)
