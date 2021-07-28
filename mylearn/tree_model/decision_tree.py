#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from collections import Counter
from collections import defaultdict
from heapq import heappush, heappop

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator


def gini(discrete_distribution):
    '''
    1 - \Sigma p^2
    :param discrete_distribution: 离散分布，如 [1,2,2,3,1]
    :return: 基尼系数，越大表示越无序
    '''
    # 频数
    frequency = list(Counter(discrete_distribution).values())
    # 频率
    sum_frequency = sum(frequency)
    frequency_ratio = [f / sum_frequency for f in frequency]
    # 计算基尼系数
    ans = 1
    for p in frequency_ratio:
        ans -= (p ** 2)
    return ans


def mse(y):
    '''
    :param y: 连续分布
    :return: 均方误差
    '''
    return np.mean(np.square(y - np.mean(y)))


def classification_leaf_value_func(y):
    return Counter(y).most_common()[0][0]


def regression_leaf_value_func(y):
    return np.mean(y)


class DataStatistic():
    class FeatureStatistic():
        def __init__(self, bin_id_to_stats, split_vals):
            self.bin_id_to_stats = bin_id_to_stats
            self.split_vals = split_vals

    def __init__(self, X, y):
        # 一个列表，长度为特征数M，每个元素为一个dict，key为特征取值所在的bin_id，value为这个bin_id对应的y的统计值
        self.feat_stat = []
        for feat_j in range(X.shape[1]):
            stat = defaultdict(list)
            for inst_i in range(X.shape[0]):
                stat[X[inst_i, feat_j]].append(y[inst_i])
            feat_vals = np.array(sorted(list(stat)))
            split_vals = (feat_vals[1:] + feat_vals[:1]) / 2



class TreeNode:
    def __init__(self, X, y, criterion, leaf_value_func, depth):
        self.depth = depth
        self.leaf_value_func = leaf_value_func
        self.criterion = criterion
        self.X = X
        self.y = y
        self.left_node = None
        self.right_node = None
        self.is_leaf = True
        self.leaf_value = self.leaf_value_func(y)

    def find_best_split_point(self):
        '''
        1. 寻找最优分裂点
        2. 并计算当前结点的信息增益值
        '''
        X = self.X
        y = self.y
        n_features = X.shape[1]
        best_feature_ix = -1
        best_threshold = -1
        best_gain = -np.inf
        impurity = self.criterion(y)
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
                y_left = y[feature <= threshold]
                y_right = y[feature > threshold]
                impurity_left = self.criterion(y_left)
                impurity_right = self.criterion(y_right)
                p_left = y_left.size / y.size
                p_right = y_right.size / y.size
                gain = impurity - (impurity_left * p_left + impurity_right * p_right)
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
        :return: 如果分裂成功，返回两个叶子结点。否则，返回None
        '''
        left_mask = (self.X[:, self.feature_ix] <= self.threshold)
        right_mask = (self.X[:, self.feature_ix] > self.threshold)
        left_node = TreeNode(
            self.X[left_mask, :], self.y[left_mask],
            self.criterion, self.leaf_value_func, self.depth + 1)
        left_node.find_best_split_point()
        right_node = TreeNode(
            self.X[right_mask, :], self.y[right_mask],
            self.criterion, self.leaf_value_func, self.depth + 1)
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


class DecisionTree(BaseEstimator):
    def __init__(self, criterion, leaf_value_func, max_depth=5):
        self.max_depth = max_depth
        self.leaf_value_func = leaf_value_func
        if criterion == "gini":
            criterion = gini
        elif criterion == "mse":
            criterion = mse
        else:
            raise NotImplementedError
        self.criterion = criterion

    def fit(self, X, y):
        self.root = TreeNode(X, y, self.criterion, self.leaf_value_func, 0)
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


class DecisionTreeClassifier(DecisionTree, ClassifierMixin):
    def __init__(self, max_depth=5):
        super(DecisionTreeClassifier, self).__init__(
            "gini", classification_leaf_value_func, max_depth=max_depth)


class DecisionTreeRegressor(DecisionTree, RegressorMixin):
    def __init__(self, max_depth=5):
        super(DecisionTreeRegressor, self).__init__(
            "mse", regression_leaf_value_func, max_depth=max_depth)
