__author__ = 'adomas'

import numpy as np
import pandas as pd
from numpy import linalg
from linearRegression import data_set, beta_hat

# non linear target function
def target_function(data):
    sign = pd.DataFrame(np.sign(data['x1'] ** 2 + data['x2'] ** 2 - 0.6))
    sign.columns = ['y']
    return pd.merge(data, sign, left_index=True, right_index=True)


# calculating in sample error
def E_in(N):
    # generating data
    data = target_function(data_set(N))
    # calculating beta
    beta = beta_hat(data)
    # subsetting data
    data_subset = data[['x0', 'x1', 'x2']].as_matrix()
    return sum(data['y'] != np.sign((data_subset*beta).sum(axis=1)))/data.shape[0]


# transforming data to non linear
def transformation(N):
    data = data_set(N)
    add_data = pd.DataFrame({'x1x2': data['x1']*data['x2'],
                  'x1sq': data['x1'] ** 2,
                  'x2sq': data['x2'] ** 2})
    return pd.merge(data, add_data, left_index=True, right_index=True)

# non linear weights
def weights_tilde():
    # generating data
    data = target_function(transformation(1000))
    # weigths vector
    return beta_hat(data)













######## answers ########

# 8 question
e_ins = []
[e_ins.append(E_in(100)) for i in np.arange(1000)]
print(np.mean(e_ins))

# 9 question
print(weights_tilde())