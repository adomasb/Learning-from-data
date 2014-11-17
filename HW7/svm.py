__author__ = 'adomas'

import numpy as np
import pandas as pd
import cvxopt
import matplotlib.pyplot as plt
import random


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


def SVM(data, N):
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


def misclassified(data, w):
    wrong = []
    for i, row in data.iterrows():
        if np.sign(np.dot(row[:3].values, w)) != row.y:
            wrong.append(i)
    return wrong


def perceptron(data):
    # init w
    w = np.array([0, 0, 0])
    counter = 0
    wrong = misclassified(data, w)
    while len(wrong) != 0:
        index = random.choice(wrong)
        row = data.irow(index)
        w = w + row[:3].values*row.y
        wrong = misclassified(data, w)
        counter += 1
    b_new, a_new = -w[:2]/w[2]
    return b_new, a_new


def Eout(N, M):

    ySum = N
    while ySum == N:
        target = TargetFunction(N)
        # training data
        data = target.data
        data_svm = data.icol([1, 2, 3])
        ySum = abs(sum(data.y))

    # out of sample data
    dataOUT = TargetFunction(M).data_set(M)
    dataOUT['y'] = np.sign(dataOUT['x2']-target.target[1]*dataOUT['x1']-target.target[0])
    dataOUT_svm = dataOUT.icol([1, 2, 3])

    # models
    b, w = SVM(data_svm, N)
    b = b.tolist()
    w = w.tolist()
    svm_coeff = np.array([-b[0]/w[1], -w[0]/w[1]])
    pla_coeff = perceptron(data)

    # svm
    svm_out = np.sign(svm_coeff[0]+np.dot(dataOUT_svm.icol([0, 1]).as_matrix(), np.array([svm_coeff[1], 1])))
    svm_err = sum(svm_out != dataOUT_svm.y)/dataOUT_svm.shape[0]
    # pla
    pla_out = np.sign(np.dot(dataOUT.icol([0, 1, 2]).as_matrix(), np.array([pla_coeff[0], pla_coeff[1], 1])))
    pla_err = sum(pla_out != dataOUT.y)/dataOUT.shape[0]

    return pla_err, svm_err


def no_sv(data):
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

    return len(data_sv)


def q10(N):
    ySum = 10
    while ySum == 10:
        target = TargetFunction(N)
        # training data
        data = target.data
        data_svm = data.icol([1, 2, 3])
        ySum = abs(sum(data.y))

    return no_sv(data_svm)


# Question 8
pla_err = []
svm_err = []
for i in np.arange(0, 1000):
    plaEout, svmEout = Eout(10, 10000)
    pla_err.append(plaEout)
    svm_err.append(svmEout)
    print(i)

sum(np.greater(svm_err, pla_err))/1000

# Question 9
pla_err = []
svm_err = []
for i in np.arange(0, 5):
    plaEout, svmEout = Eout(100, 10000)
    pla_err.append(plaEout)
    svm_err.append(svmEout)
    print(i)

sum(np.greater(svm_err, pla_err))/1000

# Question 10

q10answer = []
[q10answer.append(q10(10)) for i in np.arange(0, 1000)]


