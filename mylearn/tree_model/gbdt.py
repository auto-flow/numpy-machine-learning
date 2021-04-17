#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np


class GradientBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def preprocess_y(self, y):
        raise NotImplementedError

    def negative_gradient(self, predict, y):
        raise NotImplementedError

    def fit(self, X, y):
        # 对于分类器，可能需要做 OHE
        y = self.preprocess_y(y)
        n_classes = y.shape[1]
        # GBDT 是一个加法模型，预测值为所有的模型预测值之和
        self.estimators = []
        predict = np.zeros([n_classes])
        for i in range(self.n_estimators):
            cur_estimators = []
            target = self.negative_gradient(predict, y)
            if target.ndim == 1:
                target = target[:, np.newaxis]
