__author__ = 'adomas'

import numpy as np


def error(u, v):
    return pow((u*np.exp(v)-2*v*np.exp(-u)), 2)

def partial_u(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(np.exp(v)+2*v*np.exp(-u))




def partial_v(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))


def direction(u, v, u_init, v_init, eta):
    return u-eta*partial_u(u, v_init), v-eta*partial_v(u_init, v)

def gradient(u_init, v_init, eta):
    e = error(u_init, v_init)
    uv = 1, 1
    i = 0
    while e > pow(10, -14):
        uv = direction(uv[0], uv[1], u_init, v_init, eta)
        i += 1
        e = error(uv[0], uv[1])
        print(uv, e)
    return i






def gradient2(u_init, v_init, eta):
    u = u_init
    v = v_init
    i = 0
    e = error2(u, v)
    while e > pow(10, -14):
        u += -eta*partial_u2(u, v_init)
        v += -eta*partial_v2(u_init, v)
        e = error2(u, v)
        i += 1
    return u, v, i

