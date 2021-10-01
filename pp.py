import sys
import json
import dynamical_system
import ds_func
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import sin,cos

def func(t, x, ds):
    # sympy F
    sp_f = dynamical_system.map(sp.Matrix(x), ds.params, ds.const)
    # convert to numpy
    ret = ds_func.sp2np(sp_f).flatten()
    return ret

def main():
    # load data from json file
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    fd.close()
    # input data to constructor
    ds = dynamical_system.DynamicalSystem(json_data)

    # calculate equilibrium points
    eq = ds_func.equilibrium(ds)
    # convert to numpy
    state0 = ds_func.sp2np(eq[0,:].reshape(ds.xdim,1)).flatten()

    # calculate orbit
    duration = 0.5
    tick = 0.05
    state = solve_ivp(func, (0, duration), state0,
        method='RK45', args=(ds,),max_step=tick,
        rtol=1e-12, dense_output=True)
    print(state.y)
    plt.plot(state.y[0], state.y[1])
    plt.plot(state.y[2], state.y[3])
    plt.show()

if __name__ == '__main__':
    main()