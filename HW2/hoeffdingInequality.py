__author__ = 'adomas'

import numpy as np

# flipping coins
nu_1 = []
nu_rnd = []
nu_min = []

while len(nu_1) < 100000:
    flipped = np.random.binomial(1, 0.5, (1000, 10))
    c_1 = flipped[0]
    c_rnd = flipped[np.random.randint(0, 999)]
    c_min = flipped[flipped.sum(axis=1).argmin()]

    nu_1.append(sum(c_1)/len(c_1))
    nu_rnd.append(sum(c_rnd)/len(c_rnd))
    nu_min.append(sum(c_min)/len(c_min))


# anwsers
print(np.mean(nu_1))
print(np.mean(nu_min))
print(np.mean(nu_rnd))