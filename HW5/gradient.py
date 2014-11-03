__author__ = 'adomas'

import numpy as np
import numpy.linalg


def error(u, v):
    return pow((u*np.exp(v)-2*v*np.exp(-u)), 2)


def partial_u(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(np.exp(v)+2*v*np.exp(-u))


def partial_v(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))


def gradient(u_init, v_init, rate, limit):
    e = error(u_init, v_init)
    u, v = u_init, v_init
    i = 0
    while e > limit:
        u -= rate*partial_u(u, v_init)
        v -= rate*partial_v(u_init, v)
        u_init, v_init = u, v
        e = error(u, v)
        i += 1
    return u, v, i


def coordinate_gradient(u_init, v_init, rate, iterations):
    u, v = u_init, v_init
    for i in np.arange(iterations):
        u -= rate*partial_u(u, v)
        v -= rate*partial_v(u, v)
    return error(u, v)

# 5
print(gradient(1, 1, 0.1, pow(10, -14)))

# 6
uv = np.array(gradient(1, 1, 0.1, pow(10, -14))[0:2])

points = np.array([[1, 1], [0.713, 0.045], [0.016, 0.112], [-0.083, 0.029], [0.045, 0.024]])
print([(numpy.linalg.norm(uv - x)) for x in points])

# 7
print(coordinate_gradient(1, 1, 0.1, 30))