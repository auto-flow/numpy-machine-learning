#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-14
# @Contact    : qichun.tang@bupt.edu.cn
from nn import Linear, Tanh, MSE, SGD, Mynet
import numpy as np

np.random.seed(0)
from sklearn.preprocessing import StandardScaler
import pylab as plt

mynet = Mynet(1, 1, 9)
criterion = mynet.criterion
X = np.random.rand(100, 1) * 3
y = np.sin(X)
X = StandardScaler().fit_transform(X)
optimizer = SGD(mynet.parameters(), lr=0.1)
for epoch in range(1000):
    # running_loss = []
    # preds = []
    # for i in range(100):
    optimizer.zero_grad()
    input = X
    label = y
    pred = mynet(input)
    loss = criterion(pred, label)
    print(loss)
    mynet.backward()
    optimizer.step()
    if epoch % 20 == 0:
        plt.scatter(X.flatten(), pred)
        plt.scatter(X.flatten(), y)
        plt.show()
