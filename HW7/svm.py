__author__ = 'adomas'

import numpy as np
import pandas as pd
import cvxopt as cvxopt
import matplotlib.pyplot as plt

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


def SVM(data):
    A = cvxopt.matrix(data.y.reshape((1, N)), tc='d')
    b = cvxopt.matrix(0.0, tc='d')
    q = cvxopt.matrix(-np.transpose(np.ones(N)), tc='d')
    h = cvxopt.matrix(np.zeros(N), tc='d')
    G = cvxopt.matrix(-np.identity(N), tc='d')
    P = cvxopt.matrix(np.multiply(np.dot(data.y.reshape((N, 1)), data.y.reshape((1, N))), np.dot(data.icol([0, 1]).as_matrix(), np.transpose(data.icol([0, 1]).as_matrix()))), tc='d')

    svm_sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.array(svm_sol['x'])

    # support vectors
    data_sv = data.irow(np.where(alpha > 10e-04)[0].tolist())
    alpha_sv = alpha[np.where(alpha > 10e-04)[0].tolist()]

    w = np.dot(np.transpose(alpha_sv*data_sv.y.as_matrix().reshape((len(alpha_sv), 1))), data_sv.icol([0, 1]).as_matrix())
    b = 1/data_sv.y.as_matrix()[0] - np.dot(w, data_sv.icol([0, 1]).as_matrix()[0].reshape((2, 1)))

    return b[0], w[0]

d = 2
N = 10

target = TargetFunction(N)
data = target.data
data_svm = data.icol([1, 2, 3])
target.coeff()

plt.scatter(data_svm.x1, data_svm.x2, s=150, c=data_svm.y.as_matrix())
plt.show()

b, w = SVM(data_svm)
b = b.tolist()
w = w.tolist()
svm_coeff = np.array([-b[0]/w[1], -w[0]/w[1]])

