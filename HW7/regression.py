__author__ = 'adomas'

import numpy as np
import pandas as pd
from sklearn import linear_model

data_in = pd.read_csv('/home/adomas/Learning/Learning-from-data/HW7/in.txt', sep=';', header=None)
data_out = pd.read_csv('/home/adomas/Learning/Learning-from-data/HW7/out.txt', sep=';', header=None)


def transformation(data):
    X = pd.DataFrame({'x0': np.repeat(1, data.shape[0])})
    X['x1'] = data.icol([0])
    X['x2'] = data.icol([1])
    X['x1_2'] = pow(X.x1, 2)
    X['x2_2'] = pow(X.x2, 2)
    X['x1x2'] = X.x1*X.x2
    X['x1_x2'] = np.abs(X.x1 - X.x2)
    X['x1+x2'] = np.abs(X.x1 + X.x2)
    X['y'] = data.icol([2])
    return X

#
X_in = transformation(data_in)
X_out = transformation(data_out)

# Question 1
X_in_train = X_in[0:25]
X_in_valid = X_in[25:35]

for i in np.array([4, 5, 6, 7, 8]):
    model1 = linear_model.LinearRegression(fit_intercept=False)
    model1.fit(X_in_train.icol(np.arange(0, i)), X_in_train.y)
    y_predicted = np.sign(model1.predict(X_in_valid.icol(np.arange(0, i))))
    print((np.sum(y_predicted != X_in_valid.y.values))/X_in_valid.shape[0])

# Question 2

for i in np.array([4, 5, 6, 7, 8]):
    model1 = linear_model.LinearRegression(fit_intercept=False)
    model1.fit(X_in_train.icol(np.arange(0, i)), X_in_train.y)
    y_predicted = np.sign(model1.predict(X_out.icol(np.arange(0, i))))
    print((np.sum(y_predicted != X_out.y.values))/X_out.shape[0])

# Question 3
X_in_train = X_in[25:35]
X_in_valid = X_in[0:25]

for i in np.array([4, 5, 6, 7, 8]):
    model1 = linear_model.LinearRegression(fit_intercept=False)
    model1.fit(X_in_train.icol(np.arange(0, i)), X_in_train.y)
    y_predicted = np.sign(model1.predict(X_in_valid.icol(np.arange(0, i))))
    print((np.sum(y_predicted != X_in_valid.y.values))/X_in_valid.shape[0])

# Question 4

for i in np.array([4, 5, 6, 7, 8]):
    model1 = linear_model.LinearRegression(fit_intercept=False)
    model1.fit(X_in_train.icol(np.arange(0, i)), X_in_train.y)
    y_predicted = np.sign(model1.predict(X_out.icol(np.arange(0, i))))
    print((np.sum(y_predicted != X_out.y.values))/X_out.shape[0])

# Question 6
tmp = np.random.uniform(0, 1, (2, 100000))
np.mean(np.minimum(tmp[0], tmp[1]))