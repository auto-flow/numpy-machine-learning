#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from mylearn.tree_model import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score

X, y = load_iris(True)
tree = DecisionTreeClassifier()
tree.fit(X, y)
scores = cross_val_score(tree, X, y)
print(scores)

X, y = load_boston(True)
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
scores = cross_val_score(tree, X, y)
print(scores)
