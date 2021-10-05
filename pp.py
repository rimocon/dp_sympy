import sys
import json
import dynamical_system
import ds_func
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import sin,cos
from time import time
'''
def func(t, x, ds):
    # sympy F
    sp_f = dynamical_system.map(sp.Matrix(x), ds.params, ds.const)
    # convert to numpy
    ret = ds_func.sp2np(sp_f).flatten()
    return ret
'''
def func(t, x, ds):

    M1 = ds.const[0]
    M2 = ds.const[1]
    L1 = ds.const[2]
    L2 = ds.const[3]
    G = ds.const[4]

    a11 = M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12 = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21 = a12
    a22 = M2 * L2 * L2
    b1 = (ds.params[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - ds.params[0] * x[1])
    b2 = (ds.params[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - ds.params[1] * x[3])
    delta = a11 * a22 - a12 * a21

    ret = np.array([x[1],
                    (b1 * a22 - b2 * a12) / delta,
                    x[3],
                    (b2 * a11 - b1 * a21) / delta
                    ])
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
    state0 = ds_func.sp2np(eq[0,:].reshape(ds.xdim, 1)).flatten()

    # calculate orbit
    duration = 0.1
    tick = 0.01
    period = 0.0
    times= 0.0
    matplotinit(ds)
    while ds.running == True:
        start = time()
        state = solve_ivp(func, (0, duration), state0,
            method='RK45', args=(ds, ), max_step=tick,
            rtol=1e-12, dense_output=True)
        state_time = time() - start
        print("state計測にかかる時間",state_time)
        ds.ax.plot(state.y[0,:], state.y[1,:],
                linewidth=1, color=(0.1, 0.1, 0.3),
                ls="-")
        state0 = state.y[:, -1]
        times += state.t[-1]
        period += duration
        elapsed_time = time() - start
        print("１ループにかかる時間",elapsed_time)
        plt.pause(0.0001)  # REQIRED
    
def matplotinit(ds):
    ds.ax.set_xlim(ds.xrange)
    ds.ax.set_ylim(ds.yrange)
    plt.connect('button_press_event',
                lambda event: on_click(event, ds))
    plt.connect('key_press_event',
                lambda event: keyin(event, ds))

def on_click(event, ds):
    # left click
    if event.button == 1:
        ds.running = False
        plt.clf()
        plt.close()

def keyin(event, ds):
    if event.key == 'q':
        plt.clf()
        plt.close('all')
        print("quit")
        sys.exit()

if __name__ == '__main__':
    main()