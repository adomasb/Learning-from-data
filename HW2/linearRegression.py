__author__ = 'adomas'

import numpy as np
import pandas as pd
from numpy import linalg
import random

####### PARAMETERS ########

# number of points
N = 100
# number of points for out sample error

####### Functions #####


# generates data frame of N points
def data_set(N):
    return pd.DataFrame({'x0': np.repeat(1, N),
                         'x1': np.random.uniform(low=-1, high=1, size=N),
                         'x2': np.random.uniform(low=-1, high=1, size=N)})


# creates target function and returns intercept & slope
def target_function():
    point1 = np.random.uniform(low=-1, high=1, size=2)
    point2 = np.random.uniform(low=-1, high=1, size=2)

    a = (point1[1]-point2[1])/(point1[0]-point2[0])
    b = (point1[0]*point2[1]-point1[1]*point2[0])/(point1[0]-point2[0])
    return a, b


# assigns signs to data set by target function an returns updated data frame with y column
def signs(data_set, target_function):
    sign = pd.DataFrame(np.sign(data_set['x2']-target_function[0]*data_set['x1']-target_function[1]))
    sign.columns = ['y']
    return pd.merge(data_set, sign, left_index=True, right_index=True)


# calculates beta hat of regression
def beta_hat(data):
    X = data[['x0', 'x1', 'x2']].as_matrix()
    y = data['y'].as_matrix()
    return np.dot(np.dot(linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)


# calculates in sample error
def E_in(N):
    # target function
    f = target_function()
    # generating data
    data = signs(data_set(N), f)
    # calculating beta
    beta = beta_hat(data)
    # subsetting data
    data_subset = data[['x0', 'x1', 'x2']].as_matrix()
    return sum(data['y'] != np.sign((data_subset*beta).sum(axis=1)))/data.shape[0]


# calculates out sample error
def E_out(N, n):
    # target function
    f = target_function()
    # generating data
    data = signs(data_set(N), f)
    # calculating beta
    beta = beta_hat(data)

    # out of sample data
    data_out = signs(data_set(n), f)
    # subsetting data
    data_out_subset = data_out[['x0', 'x1', 'x2']].as_matrix()
    return sum(data_out['y'] != np.sign((data_out_subset*beta).sum(axis=1)))/data_out.shape[0]



# misclassified array function: returns array of misclassified points indeces
def misclassified(data, w):
    wrong = []
    for i, row in data.iterrows():
        if np.sign(np.dot(row[:3].values, w)) != row.y:
            wrong.append(i)
    tmp = row
    return wrong


# does perceptron and returns count of itterations
def perceptron(data, w):
    counter = 0
    wrong = misclassified(data, w)
    while len(wrong) != 0:
        index = random.choice(wrong)
        row = data.irow(index)
        w = w + row[:3].values*row.y
        wrong = misclassified(data, w)
        counter += 1
    return counter


# calculates weights with linear regression and perceptron finalizes them
def lr_perceptron(N):
    # target function
    f = target_function()
    # generating data
    data = signs(data_set(N), f)
    # calculating beta
    beta = beta_hat(data)
    # perceptron
    return perceptron(data, beta)


###### Answers ##########

# 5 question answer
e_ins = []
[e_ins.append(E_in(100)) for i in np.arange(1000)]
print(np.mean(e_ins))

# 6 question answer
e_outs = []
[e_outs.append(E_out(100, 1000)) for i in np.arange(1000)]
print(np.mean(e_outs))

# 7 question answer
counts = []
[counts.append(lr_perceptron(10)) for i in np.arange(1000)]
print(np.mean(counts))