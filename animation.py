import sys
import json
import dynamical_system
import ds_func
from numpy import sin, cos
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt


G = 9.8     # 重力加速度 [m/s^2]
L1 = 1.0    # 単振り子1の長さ [m]
L2 = 1.0    # 単振り子2の長さ [m]
M1 = 1.0    # おもり1の質量 [kg]
M2 = 1.0    # おもり2の質量 [kg]

# 運動方程式
def func(t, x, p, c):
    M1 = c[0]
    M2 = c[1]
    L1 = c[2]
    L2 = c[3]
    G = c[4]


    a11 = M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12 = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21 = a12
    a22 = M2 * L2 * L2
    b1 = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2 = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    delta = a11 * a22 - a12 * a21

    ret = np.array([
        x[1],
        (b1 * a22 - b2 * a12) / delta,
        x[3],
        (b2 * a11 - b1 * a21) / delta
    ])
    return ret

def derivs(t, state):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

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
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy const
    c = ds_func.sp2np(ds.const).flatten()

# ジェネレータ
def gen():
    for tt, th1, th2 in zip(t,y[0,:], y[2,:]):
        x1 = L1*sin(th1)
        y1 = -L1*cos(th1)
        x2 = L2*sin(th2) + x1
        y2 = -L2*cos(th2) + y1
        yield tt, x1, y1, x2, y2
'''
fig, ax = plt.subplots()
ax.set_xlim(-(L1+L2), L1+L2)
ax.set_ylim(-(L1+L2), L1+L2)
ax.set_aspect('equal')
ax.grid()

locus, = ax.plot([], [], 'r-', linewidth=2)
line, = ax.plot([], [], 'o-', linewidth=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

xlocus, ylocus = [], []
def animate(data):
    t, x1, y1, x2, y2 = data
    xlocus.append(x2)
    ylocus.append(y2)

    locus.set_data(xlocus, ylocus)
    line.set_data([0, x1, x2], [0, y1, y2])
    time_text.set_text(time_template % (t))

ani = FuncAnimation(fig, animate, gen, interval=50, repeat=True)

plt.show()

# ani.save('double_pendulum.gif', writer='pillow', fps=15)
'''