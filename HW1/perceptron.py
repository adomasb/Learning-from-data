import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

####### PARAMETERS ########

# number of points
N = 10

######## DATA & TARGET FUNCTION #####

# data
X = pd.DataFrame({'x0': np.repeat(1, N),
                  'x1': np.random.uniform(low=-1, high=1, size=N),
                  'x2': np.random.uniform(low=-1, high=1, size=N)})

# target function
point1 = np.random.uniform(low=-1, high=1, size=2)
point2 = np.random.uniform(low=-1, high=1, size=2)

a = (point1[1]-point2[1])/(point1[0]-point2[0])
b = (point1[0]*point2[1]-point1[1]*point2[0])/(point1[0]-point2[0])

# points for ploting
y = a*np.array([1, -1])+b


def target(X, a, b):
    sign = pd.DataFrame(np.sign(X['x2']-a*X['x1']-b))
    sign.columns = ['y']
    return pd.merge(X, sign, left_index=True, right_index=True)

# final data
X = target(X, a, b)

# plot
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.scatter(point1[0], point1[1], color='black')
plt.scatter(point2[0], point2[1], color='black')
plt.plot([1, -1], y, color='g')
plt.scatter(x=X.x1, y=X.x2, c=np.where(X.y == 1, 'r', 'b'))
plt.show()

######## ALGORITHM IMPLEMENTATION ######

# misclassified array function
def misclassified(X, w):
    wrong = []
    for i, row in X.iterrows():
        if np.sign(np.dot(row[:3].values, w)) != row.y:
            wrong.append(i)
    tmp = row
    return wrong

# init w
w = np.array([0, 0, 0])

# misclassified indeces
wrong = misclassified(X, w)

# counter
counter = 0

while len(wrong) != 0:
    index = random.choice(wrong)
    row = X.irow(index)
    w = w + row[:3].values*row.y
    wrong = misclassified(X, w)
    counter += 1

# new coefs
b_new, a_new = -w[:2]/w[2]

# testing point for new separating line
y_new = a_new*np.array([1, -1])+b_new

# plot
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.scatter(point1[0], point1[1], color='black')
plt.scatter(point2[0], point2[1], color='black')
plt.plot([1, -1], y, color='g')
plt.plot([1, -1], y_new, color='red')
plt.scatter(x=X.x1, y=X.x2, c=np.where(X.y == 1, 'r', 'b'))
plt.show()

######### ANSWERING QUESTIONS ########

# 7 question
counterList = []

for i in np.arange(1000):
    N = 10

    # data
    X = pd.DataFrame({'x0': np.repeat(1, N),
                  'x1': np.random.uniform(low=-1, high=1, size=N),
                  'x2': np.random.uniform(low=-1, high=1, size=N)})

    # target function
    point1 = np.random.uniform(low=-1, high=1, size=2)
    point2 = np.random.uniform(low=-1, high=1, size=2)

    a = (point1[1]-point2[1])/(point1[0]-point2[0])
    b = (point1[0]*point2[1]-point1[1]*point2[0])/(point1[0]-point2[0])

    X = target(X, a, b)

    w = np.array([0, 0, 0])

    wrong = misclassified(X, w)


    counter = 0

    while len(wrong) != 0:
        index = random.choice(wrong)
        row = X.irow(index)
        w = w + row[:3].values*row.y
        wrong = misclassified(X, w)
        counter += 1

    counterList.append(counter)
