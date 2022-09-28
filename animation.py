import sys
import json
import dynamical_system
import ds_func
import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# プロット用設定
plt.rcParams["font.family"] = "Nimbus Roman"    #全体のフォントを設定
plt.rcParams['text.usetex'] = True              #描画にTeXを利用
plt.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
                                          \usepackage{amssymb}
                                          \usepackage[T1]{fontenc}
                                          \usepackage{bm}
                                          \usepackage{xcolor}
                                          '''
plt.rcParams["figure.autolayout"] = False       #レイアウト自動調整をするかどうか
plt.rcParams["font.size"] = 24                  #フォントの大きさ
plt.rcParams["xtick.direction"] = "in"          #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"          #y軸の目盛線を内向きへ
# plt.rcParams["xtick.minor.visible"] = True      #x軸補助目盛りの追加
# plt.rcParams["ytick.minor.visible"] = True      #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.0         #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.0         #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 0.5         #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 0.5         #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 20           #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 20          #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 10            #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 10            #y軸補助目盛り線の長さ
plt.rcParams["xtick.major.pad"] = 16             #x軸と目盛数値のマージン
plt.rcParams["ytick.major.pad"] = 16             #y軸と目盛数値のマージン
plt.rcParams["axes.linewidth"] = 2            #囲みの太さ

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

def solver(ds):
    # solver
    ds.state = solve_ivp(func, (0, ds.duration), ds.state0, 
        method='RK45', t_eval = ds.eval, args = (ds.p, ds.c), 
        rtol=1e-12, vectorized = True)

def animate(i):
    thisx = [0, ds.x1[i], ds.x2[i]]
    thisy = [0, ds.y1[i], ds.y2[i]]

    if i == 0:
        ds.history_x.clear()
        ds.history_y.clear()
    ds.history_x.appendleft(thisx[2])
    ds.history_y.appendleft(thisy[2])

    ds.line.set_data(thisx, thisy)
    ds.trace.set_data(ds.history_x, ds.history_y)
    ds.time_text.set_text(ds.time_template % (ds.state.t[i]))
    return ds.line, ds.trace, ds.time_text

# generator
def gen(ds):
    ds.x1 = ds.c[0] * cos(ds.state.y[0,:])
    ds.y1 = ds.c[0] * sin(ds.state.y[0,:])
    ds.x2 = ds.x1 + ds.c[1] * cos(ds.state.y[0,:] + ds.state.y[2,:])
    ds.y2 = ds.y1 + ds.c[1] * sin(ds.state.y[0,:] + ds.state.y[2,:])

def set(ds):
    # input data to constructor
    eq = ds_func.equilibrium(ds)
    # convert to numpy
    ds.eq = ds_func.sp2np(eq)
    ds.state0 = ds.eq[ds.x_ptr,:].flatten()
    # ds.state0 = np.array([1,1,1,1])
    print(ds.state0)
    # import numpy parameter
    ds.p = ds_func.sp2np(ds.params).flatten()

    # import numpy constant
    ds.c = ds_func.sp2np(ds.const).flatten()

   
    ds.ax.set_xlim(-(ds.c[0]+ds.c[1]),ds.c[0]+ds.c[1])
    ds.ax.set_ylim(-(ds.c[0]+ds.c[1]),ds.c[0]+ds.c[1])
    ds.ax.set_xticks([-(ds.c[0]+ds.c[1]),-(ds.c[0]+ds.c[1])/2, 0, (ds.c[0]+ds.c[1])/2, ds.c[0]+ds.c[1]])
    ds.ax.set_yticks([-(ds.c[0]+ds.c[1]),-(ds.c[0]+ds.c[1])/2, 0, (ds.c[0]+ds.c[1])/2, ds.c[0]+ds.c[1]])
    ds.ax.set_aspect('equal')
    s = ""
    eq= ""
    cnt = 0
    for key in ds.p:
        s += f"p{cnt:d} {key:.4f} "
        cnt += 1
    cnt = 0
    for key in ds.state0:
        eq += f"x{cnt:d}0 {key:.4f} "
        cnt += 1
    cnt = 0
    title = s + "\n" + eq
    plt.title(title, color='b')
    ds.ax.grid()
    
def keyin(event, ds):
    if event.key == 'q':
        plt.cla()
        plt.close('all')
        print("quit")
        sys.exit()
    elif event.key == 'x':
        plt.cla()
        ds.x_ptr += 1
        if(ds.x_ptr >= ds.xdim):
            ds.x_ptr = 0
        set(ds)
        locus(ds)
        solver(ds)
        gen(ds)
        ds.ani.frame_seq = ds.ani.new_frame_seq()
        plt.show()

    elif event.key == 'p':
        ds.p_ptr += 1
        if(ds.p_ptr >= 4):
            ds.p_ptr = 0
        print(f"changable paramter: {ds.p_ptr}")

    elif event.key == 'up':
        plt.cla()
        print(f"change paramter[{ds.p_ptr}]")
        print(f"before value:{ds.params[ds.p_ptr]}")
        ds.params[ds.p_ptr] += ds.d
        print(f"after value:{ds.params[ds.p_ptr]}")
        set(ds)
        locus(ds)
        solver(ds)
        gen(ds)
        ds.ani.frame_seq = ds.ani.new_frame_seq()
    elif event.key == 'down':
        plt.cla()
        print(f"change paramter[{ds.p_ptr}]")
        print(f"before value:{ds.params[ds.p_ptr]}")
        ds.params[ds.p_ptr] -= ds.d
        print(f"after value:{ds.params[ds.p_ptr]}")
        set(ds)
        locus(ds)
        solver(ds)
        gen(ds)
        ds.ani.frame_seq = ds.ani.new_frame_seq()
    elif event.key == ' ':
        print("redraw")
        plt.cla()
        set(ds)
        locus(ds)
        solver(ds)
        gen(ds)
        ds.ani.frame_seq = ds.ani.new_frame_seq()
    elif event.key == 'right':
        ds.d = ds.d * 10
        print("change increment/decrement scale:",ds.d)
        
    
    elif event.key == 'left':
        ds.d = ds.d / 10
        print("change increment/decrement scale:",ds.d)
   
def locus(ds):

    ds.line, = ds.ax.plot([], [], 'o-', lw=2)
    ds.trace, = ds.ax.plot([], [], '.-', lw=1, ms=2)
    ds.time_template = 'time = %.1fs'
    ds.time_text = ds.ax.text(0.05, 0.9, '', transform=ds.ax.transAxes)
    ds.history_x, ds.history_y = deque(maxlen=ds.history_len), deque(maxlen=ds.history_len)

# load data from json file
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} filename")
fd = open(sys.argv[1], 'r')
json_data = json.load(fd)
fd.close()

ds = dynamical_system.DynamicalSystem(json_data)
ds.x_ptr = 3
ds.p_ptr = 0
ds.duration = 40
ds.tick = 0.01
ds.eval = np.arange(0, ds.duration, ds.tick)
ds.history_len = 500


set(ds)
# calculate orbit
solver(ds)
# graph
plt.connect('key_press_event',
    lambda event: keyin(event, ds))
locus(ds)
gen(ds)
ds.ani = FuncAnimation(ds.fig, animate, len(ds.state.t), interval=ds.tick * 1000,blit = True, repeat = False)
    # ani.save('double_pendulum.gif', writer='pillow', fps=15)
plt.show()
