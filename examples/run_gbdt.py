#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from mylearn.tree_model.gbdt import GradientBoostRegressor, GradientBoostClassifier
from mylearn.tree_model import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

X, y = load_iris(True)
gbdt = GradientBoostClassifier(n_estimators=20, learning_rate=0.5, max_depth=3)
scores = cross_val_score(gbdt, X, y)
print(scores)

X, y = load_boston(True)
gbdt = GradientBoostRegressor(n_estimators=20, learning_rate=0.5, max_depth=3)
scores = cross_val_score(gbdt, X, y)
print(scores)
