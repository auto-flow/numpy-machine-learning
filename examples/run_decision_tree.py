#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from mylearn.tree_model import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from time import time


def run_score(load_func, klass, anotation):
    start_time = time()
    X, y = load_func(True)
    tree = klass(max_depth=5)
    tree.fit(X, y)
    scores = cross_val_score(tree, X, y)
    cost_time = time() - start_time
    print(f"{anotation} {klass.__name__}:")
    print(scores)
    print(f"cost time = {cost_time:.3f}")
    print()


run_score(load_iris, DecisionTreeClassifier, "My")
run_score(load_iris, SklearnDecisionTreeClassifier, "Sklearn")
run_score(load_boston, DecisionTreeRegressor, "My")
run_score(load_boston, SklearnDecisionTreeRegressor, "Sklearn")
