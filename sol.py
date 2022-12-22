import sys
import json
import dynamical_system
import ds_func
import dp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import sin,cos


# # プロット用設定
# plt.rcParams["font.family"] = "Nimbus Roman"    #全体のフォントを設定
# plt.rcParams['text.usetex'] = True              #描画にTeXを利用
# plt.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
#                                           \usepackage{amssymb}
#                                           \usepackage[T1]{fontenc}
#                                           \usepackage{bm}
#                                           \usepackage{xcolor}
#                                           '''
# plt.rcParams["figure.autolayout"] = False       #レイアウト自動調整をするかどうか
# plt.rcParams["font.size"] = 24                  #フォントの大きさ
# plt.rcParams["xtick.direction"] = "in"          #x軸の目盛線を内向きへ
# plt.rcParams["ytick.direction"] = "in"          #y軸の目盛線を内向きへ
# plt.rcParams["xtick.minor.visible"] = True      #x軸補助目盛りの追加
# plt.rcParams["ytick.minor.visible"] = True      #y軸補助目盛りの追加
# plt.rcParams["xtick.major.width"] = 1.0         #x軸主目盛り線の線幅
# plt.rcParams["ytick.major.width"] = 1.0         #y軸主目盛り線の線幅
# plt.rcParams["xtick.minor.width"] = 0.5         #x軸補助目盛り線の線幅
# plt.rcParams["ytick.minor.width"] = 0.5         #y軸補助目盛り線の線幅
# plt.rcParams["xtick.major.size"] = 20           #x軸主目盛り線の長さ
# plt.rcParams["ytick.major.size"] = 20          #y軸主目盛り線の長さ
# plt.rcParams["xtick.minor.size"] = 10            #x軸補助目盛り線の長さ
# plt.rcParams["ytick.minor.size"] = 10            #y軸補助目盛り線の長さ
# plt.rcParams["xtick.major.pad"] = 16             #x軸と目盛数値のマージン
# plt.rcParams["ytick.major.pad"] = 16             #y軸と目盛数値のマージン
# plt.rcParams["axes.linewidth"] = 2            #囲みの太さ


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
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    fd.close()
    # input data to constructor
    ds = dynamical_system.DynamicalSystem(json_data)
    
    Eigen(ds)
    ##################################################
    # solve
    ##################################################
    duration = [0,20] 
    tick = 0.01
    t_eval = np.arange(duration[0],duration[1],tick)
    print(t_eval)
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    # calculate equilibrium points
    eq = ds_func.equilibrium(ds)
    # convert to numpy
    ds.eq = ds_func.sp2np(eq)

    ds.state0 = ds.eq[0,:].flatten()
    
    state = solve_ivp(func, duration, ds.x0,
        method='RK45', args = (p, c), t_eval = t_eval,
        rtol=1e-12)
    print(state.t)
    label = "stable"
    color = (1.0, 0.0, 0.0)
    for i in range(4):
        a = solve_ivp(func, duration, ds.xa[i,:],
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12)
        b = solve_ivp(func, duration, ds.xb[i,:],
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12)
        if i >= 2:
            label = "unstable"
            color = (0.0, 0.0, 1.0)
        ds.ax2.plot(a.y[0,:], a.y[1,:],
            linewidth=1, color = color,
            label = label,ls="-")
        ds.ax2.plot(b.y[0,:], b.y[1,:],
            linewidth=1, color = color, 
            label = label,ls="-")

    ##################################################
    # plot
    ##################################################
    show_param(ds)
    ds.ax2.set_xlim(ds.xrange)
    ds.ax2.set_ylim(ds.yrange)
    ds.ax2.grid(c='gainsboro', ls='--', zorder=9)


    # ds.ax2.plot(state.y[0,:], state.y[1,:],
    #         linewidth=1,color = (0.0, 0.0, 0.0),
    #         label = "state0",ls="-")

    ds.ax2.legend()
    plt.show()

def show_param(ds):
    s = ""
    p = ""
    params = ds_func.sp2np(ds.params).flatten().tolist()
    x0 = ds.state0.flatten().tolist()
    for i in range(len(params)):
        s += f"x{i}:{x0[i]:.5f},"
        p += f"p{i}:{params[i]:.4f},"
    plt.title(s+"\n"+p, color = 'blue')

def Eigen(ds):
    eq = ds_func.equilibrium(ds)
    ds.x0 = ds_func.sp2np(eq)
    ds.x0 = ds.x0[0,:]
    print("x0\n",ds.x0)
    # for i in range(4):
    #     eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)

    eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, 0)
    print("eigenvalue\n", eig)
    eig_vr = eig_vr * (-1)
    print("eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xa = ds.x0 + delta

    eig_vr = eig_vr * (-1)
    print("inverse_eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xb = ds.x0 + delta


    print("x0", ds.x0)
    print("xa", ds.xa)
    print("xb", ds.xb)

if __name__ == '__main__':
    main()