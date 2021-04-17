#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-17
# @Contact    : qichun.tang@bupt.edu.cn
from mylearn.tree_model.gbdt import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as SklearnGradientBoostingRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import cross_val_score
from time import time


def run_score(load_func, klass, anotation, params):
    start_time = time()
    X, y = load_func(True)
    tree = klass(**params)
    tree.fit(X, y)
    scores = cross_val_score(tree, X, y)
    cost_time = time() - start_time
    print(f"{anotation} {klass.__name__}:")
    # print(f"params = {params}")
    print(scores)
    print(f"cost time = {cost_time:.3f}")
    print()


for params in [
    dict(n_estimators=10, learning_rate=0.1),
    dict(n_estimators=10, learning_rate=1),
    dict(n_estimators=20, learning_rate=0.5),
]:
    print("#" * 50)
    print(params)
    print("#" * 50)
    run_score(load_iris, GradientBoostingClassifier, "My", params)
    run_score(load_iris, SklearnGradientBoostingClassifier, "Sklearn", params)
    run_score(load_boston, GradientBoostingRegressor, "My", params)
    run_score(load_boston, SklearnGradientBoostingRegressor, "Sklearn", params)
