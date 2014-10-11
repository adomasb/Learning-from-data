__author__ = 'adomas'

import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt

####### PARAMETERS ########

# number of points
N = 100

######## DATA & TARGET FUNCTION #####

# generates data
def data_set(N):
    # data
    data = pd.DataFrame({'x0': np.repeat(1, N),
                         'x1': np.random.uniform(low=-1, high=1, size=N),
                         'x2': np.random.uniform(low=-1, high=1, size=N)})

    # target function
    point1 = np.random.uniform(low=-1, high=1, size=2)
    point2 = np.random.uniform(low=-1, high=1, size=2)

    a = (point1[1]-point2[1])/(point1[0]-point2[0])
    b = (point1[0]*point2[1]-point1[1]*point2[0])/(point1[0]-point2[0])

    # final data
    sign = pd.DataFrame(np.sign(data['x2']-a*data['x1']-b))
    sign.columns = ['y']

    return pd.merge(data, sign, left_index=True, right_index=True)

# plotting data
data = data_set(100)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.scatter(x=data.x1, y=data.x2, c=np.where(data.y == 1, 'r', 'b'))
plt.show()

######## ALGORITHM IMPLEMENTATION ######

# separating data
data = data_set(100)
X = data[['x0', 'x1', 'x2']].as_matrix()
y = data['y'].as_matrix()

# beta hat
def beta_hat(X, y):
    beta_hat = np.dot(np.dot(linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)
    return beta_hat

# points for plotting
beta = beta_hat(X, y)
points_beta = -beta[0]/beta[2] - beta[1]/beta[2] * np.array([1, -1])

# plotting
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.plot([1, -1], points_beta, color='yellow')
plt.scatter(x=data.x1, y=data.x2, c=np.where(data.y == 1, 'r', 'b'))
plt.show()

# calculates in sample error
def E_in(data):
    data = data_set(100)
    X = data[['x0', 'x1', 'x2']].as_matrix()
    y = data['y'].as_matrix()
    beta = beta_hat(X, y)
    data_subset = data[['x0', 'x1', 'x2']].as_matrix()
    return sum(data['y']!=np.sign((data_subset*beta).sum(axis=1)))/data.shape[0]

# 5 question answer
e_ins = []
[e_ins.append(E_in(data_set(100))) for i in np.arange(1000)]
print(np.mean(e_ins))

#
data = data_set(100)
X = data[['x0', 'x1', 'x2']].as_matrix()
y = data['y'].as_matrix()
beta = beta_hat(X, y)







