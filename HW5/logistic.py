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

