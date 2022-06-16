import sys
import json
import dynamical_system
import ds_func
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from sympy import sin,cos
from numpy import sin,cos


'''
# if use sympy matrix
def func(t, x, ds):
    # sympy F
    sp_f = dynamical_system.map(sp.Matrix(x), ds.params, ds.const)
    # convert to numpy
    ret = ds_func.sp2np(sp_f).flatten()
    return ret
'''
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
    ds.eq = ds_func.sp2np(eq)
    ds.state0 = ds.eq[0,:].flatten()
    # calculate orbit
    duration = 0.5
    tick = 0.05
    matplotinit(ds)
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    show_params(ds)
    while ds.running == True:
        state = solve_ivp(func, (0, duration), ds.state0,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True) 
        
        ds.ax.plot(state.y[0,:], state.y[1,:],
                linewidth=1, color=(0.1, 0.1, 0.3),
                ls="-")
        #ds.ax.plot(state.y[2,:], state.y[3,:],
         #       linewidth=1, color=(0.3, 0.1, 0.1),
         #       ls="-")
        ds.state0 = state.y[:, -1]
        plt.pause(0.001)  # REQIRED

def matplotinit(ds):
    redraw(ds)
    plt.connect('button_press_event',
                lambda event: on_click(event, ds))
    plt.connect('key_press_event',
                lambda event: keyin(event, ds))

def redraw(ds):
    show_params(ds)
    ds.ax.set_xlim(ds.xrange)
    ds.ax.set_ylim(ds.yrange)
    ds.ax.grid(c='gainsboro', ls='--', zorder=9)

def eq_change(ds):
    ds.x_ptr += 1
    if(ds.x_ptr >= ds.xdim):
        ds.x_ptr = 0
    ds.state0 = ds.eq[ds.x_ptr, :].flatten()

def on_click(event, ds):
    #left click
    if event.xdata == None or event.ydata == None:
        return
    plt.cla()
    redraw(ds)
    if ds.dim_ptr <= 0:
        ds.state0[0] = event.xdata
        ds.state0[1] = event.ydata
    if ds.dim_ptr >= 1:
        ds.state0[2] = event.xdata
        ds.state0[3] = event.ydata

def state_reset(ds):
    ds.state0 = ds.eq[ds.x_ptr,:].flatten()

def keyin(event, ds):
    if event.key == 'q':
        ds.running = False
        plt.clf()
        plt.close('all')
        print("quit")
        sys.exit()
    elif event.key == ' ':
        ds.ax.cla()
        redraw(ds)
        state_reset(ds)
    elif event.key == 'x':
        ds.ax.cla()
        eq_change(ds)
        redraw(ds)
    elif event.key == 'c':
        ds.dim_ptr += 1
        if(ds.dim_ptr >= 2):
            ds.dim_ptr = 0
        if(ds.dim_ptr <= 0):
            print("change on_click dimension to theta_1, theta_1 dot")
        elif(ds.dim_ptr >= 1):
            print("change on_click dimension to theta_2, theta_2 dot")
    elif event.key == 'p':
        # change parameter
        print("change parameter")
    elif event.key == 's':
        print("now writing...")
        pdf = PdfPages('snapshot.pdf')
        pdf.savefig()
        pdf.close()
        print("done.")
    

def show_params(ds):
    s = ""
    p = ""
    params = ds_func.sp2np(ds.params).flatten().tolist()
    x0 = ds.state0.flatten().tolist()
    for i in range(len(params)):
        s += f"x{i}:{x0[i]:.5f},"
        p += f"p{i}:{params[i]:.4f},"
    plt.title(s+"\n"+p, color = 'blue')

if __name__ == '__main__':
    main()