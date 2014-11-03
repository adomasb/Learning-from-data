__author__ = 'adomas'

import numpy as np
import pandas as pd

class TargetFunction(object):
    def __init__(self, n):
        self.n = n
        self.target = self.coeff()
        self.generate_data = self.data_set(self.n)
        self.data = self.signs(self.generate_data, self.target)

    def coeff(self):
        point1 = np.random.uniform(low=-1, high=1, size=2)
        point2 = np.random.uniform(low=-1, high=1, size=2)

        a = (point1[1]-point2[1])/(point1[0]-point2[0])
        b = (point1[0]*point2[1]-point1[1]*point2[0])/(point1[0]-point2[0])
        return a, b

    def data_set(self, n):
        return pd.DataFrame({'x0': np.repeat(1, n),
                             'x1': np.random.uniform(low=-1, high=1, size=n),
                             'x2': np.random.uniform(low=-1, high=1, size=n)})

    def signs(self, data_set, target_function):
        sign = pd.DataFrame(np.sign(data_set['x2']-target_function[0]*data_set['x1']-target_function[1]))
        sign.columns = ['y']
        return pd.merge(data_set, sign, left_index=True, right_index=True)


def randomize(data):
    return data.reindex(np.random.permutation(data.index))


def direction(point, w):
    x = point[0:3].as_matrix()
    y = point[3]
    return -(x*y)/(1+np.exp(y*np.dot(w, x)))


def epoch(data, w, rate):
    data = randomize(data)
    for i in np.arange(data.shape[0]):
        w = w - rate * direction(data.ix[i], w)
    return w


def SGD(data, w_init, rate, limit):
    w = epoch(data, w_init, rate)
    i = 0
    while np.sqrt(sum(pow(w - w_init, 2))) >= limit:
        w_init = w
        w = epoch(data, w_init, rate)
        i += 1
    return w, i


def crossentropy(data, w):
    X = data.icol([0, 1, 2]).as_matrix()
    Y = data.y.as_matrix()
    return np.mean(np.log(1+np.exp(-Y*np.dot(X, w))))


# 8
Ein = []

for i in np.arange(100):
    target = TargetFunction(1100)
    data = target.data[0:100]
    test = target.data[101:1000]

    w_init = np.array([0, 0, 0])
    rate = 0.01
    limit = 0.01

    w = SGD(data, w_init, rate, limit)[0]
    Ein.append(crossentropy(test, w))

print(np.mean(Ein))

# 9
iteration = []

for i in np.arange(100):
    target = TargetFunction(100)
    data = target.data
    w_init = np.array([0, 0, 0])
    rate = 0.01
    limit = 0.01

    iteration.append(SGD(data, w_init, rate, limit)[1])

print(np.mean(iteration))