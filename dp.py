from re import A
import sys
import json
import dynamical_system
import ds_func
import variation
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import sin,cos
from scipy import linalg


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
    # import numpy parameter
    ds.p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    ds.c = ds_func.sp2np(ds.const).flatten()



    ds.x0 = Eq_check(ds)
    Eigen(ds)
    solve(ds)
    # ここのパラメタの入れ込み考え
    # 多分vpとかを入れることになる
    p = np.array([0,100,0,0])
    z = np.concatenate([ds.x_alpha,ds.x_omega,p])
    z = sp.Matrix(z.reshape(12,1))
    print("z",z)
    sym_F = ds_func.newton_F(ds.sym_z,ds)
    print("sym_F= ",sym_F)
    sym_J = sym_F.jacobian(ds.sym_z)
    print("sym_J= ",sym_J)
    F = sym_F.subs(ds.sym_z,z)
    J = sym_J.subs(ds.sym_z,z)
    F = ds_func.sp2np(F)
    J = ds_func.sp2np(J)
    print("F(subs)= ",F)
    print("J(subs)= ",J)


def Eigen(ds):
    eq = ds_func.equilibrium(ds)
    x0 = ds_func.sp2np(eq)
    x0 = x0[0,:]
    print("x0\n",ds.x0)
    # for i in range(4):
    #     eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)

    eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, 0)
    print("eigenvalue\n", eig)
    ds.mu_alpha = eig[1]
    ds.mu_omega = eig[2]
    eig_vr = eig_vr * (-1)
    print("eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xa = x0 + delta

    eig_vr = eig_vr * (-1)
    print("inverse_eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xb = x0 + delta

    ####ここは８本から２本選択
    ds.x_alpha = ds.xa[1,:]
    ds.x_omega = ds.xa[2,:]
    print("x0", x0)
    print("x_alpha", ds.x_alpha)
    print("x_omega", ds.x_omega)

def Eq_check(ds):
    eq = ds_func.equilibrium(ds)
    vp = eq[0,:].T
    vp = sp.Matrix(vp)
    print("eq=",vp)

    # for i in range(ds.xdim):
    #     F = ds.F.subs([(ds.sym_x, vp[0,:].T), (ds.sym_p, ds.params)])
    #     print(F'eq{i} = {F}')

    J = ds.F.jacobian(ds.sym_x)
    for i in range(ds.iter_max):
        F = ds.F.subs([(ds.sym_x, vp), (ds.sym_p, ds.params)])
        J = J.subs([(ds.sym_x, vp), (ds.sym_p, ds.params)])
        F = ds_func.sp2np(F)
        J = ds_func.sp2np(J)
        # print(F'eq{1} = {F}')
        # print(J)

        dif = abs(np.linalg.norm(F))
        print("dif=",dif)
        if dif < ds.eps:
            print("success!!!")
            print("solve vp = ",vp)
            return vp
        if dif > ds.explode:
            print("Exploded")
            exit()
            # vn = xk+1
            # print("vp=",vp)
        vn = np.linalg.solve(J,-F) + vp
        print("i=",i)
        print("vn=",vn)
        vp = vn
        # if vn[5] > 1.0:
        #   print("B0 is too high")




def solve(ds):
    ####solve####
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    # convert initial value to numpy
    # ds.state0 = ds_func.sp2np(ds.x0).flatten()
    
    vari_ini = np.eye(4).reshape(16,)
    vari_param_ini = np.zeros(4)
    print("initial param",vari_param_ini)
    vari_ini = np.concatenate([vari_ini,vari_param_ini])
    print("initial vari",vari_ini)
    ds.x0_p = np.concatenate([ds.x_alpha,vari_ini])
    ds.x0_m = np.concatenate([ds.x_omega,vari_ini])
    print("initial value",ds.x0_p)

    # for plus
    # ds.state_p = solve_ivp(variation.func, ds.duration, ds.state0,
    #     method='RK45', args = (p, c), t_eval = ds.t_eval,
    #     rtol=1e-12)
    ## argsのところもvpに変える必要がある
    ds.state_p = solve_ivp(variation.func, ds.duration, ds.x0_p,
        method='RK45', args = (p, c),
        rtol=1e-12)
    print("y_p",ds.state_p.y)
    print("y0 - y4",ds.state_p.y[0:4,-1])
    ## for minus
    # print("duration",ds.duration_m)
    # print("eval",ds.t_eval_m)
    # ds.state_m = solve_ivp(variation.func, ds.duration_m, ds.x0_m,
    #     method='RK45', args = (p, c), 
    #     rtol=1e-12)
    # # print("t_m",ds.state_m.t)
    # print("y_m",ds.state_m.y)
    # a = ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
    # print("test",a)



if __name__ == '__main__':
    main()