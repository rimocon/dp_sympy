from re import A
import sys
import json
from this import d
import dynamical_system
import ds_func
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
    Eq_check(ds)
    ds.duration = 40
    ds.tick = 0.01
    ds.eval = np.arange(0, ds.duration, ds.tick)
    Alpha(ds)
    ds.state0 = ds.x_alpha[0,:].flatten()
    print(ds.state0)
    # solver
    ds.state = solve_ivp(func, (0, ds.duration), ds.state0, 
        method='RK45', t_eval = ds.eval, args = (ds.p, ds.c), 
        rtol=1e-12, vectorized = True)
    print(ds.state.t)
    

    '''
    ##### for all eqpoints#####
    for i in range(ds.xdim):
        delta_vec = eig_vr[:,i] * ds.delta
        ds.x_alpha = np_eq + delta_vec
    '''
    print("x_alpha\n",ds.x_alpha)
    print("x_omega\n",ds.x_omega)
    Eq_check(ds)
    # test = Condition(ds)
    # # print("条件式テスト",test)
    # jac_test = jac(ds)
    # # print("ヤコビアンテスト",jac_test)
def Alpha(ds):
    eq = ds_func.equilibrium(ds)
    ds.x0 = ds_func.sp2np(eq)
    print("x0\n",ds.x0)
    for i in range(4):
        eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)
        ds.mu_alpha = eig[0]
        ds.mu_omega = eig[1]

        print("eigenvalue\n", eig)
    eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, 0)
    ds.mu_alpha = eig[0]
    ds.mu_omega = eig[1]


    print("mu_alphaに対する固有ベクトル",eig_vr[:,0])
    print("mu_omegaに対する固有ベクトル",eig_vr[:,1])
    delta_alpha = eig_vr[:,0] * ds.delta
    delta_omega = eig_vr[:,1] * ds.delta
    print("delta_alpha", delta_alpha)
    x_alpha = ds.x0 + delta_alpha
    x_omega = ds.x0 + delta_omega

    ds.x_alpha = x_alpha
    ds.x_omega = x_omega
def Eigen(ds):
    eq = ds_func.equilibrium(ds)
    ds.x0 = ds_func.sp2np(eq)
    print("x0\n",ds.x0)

    # for i in range(4):
    #     eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)

    eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, 0)
    print("eigenvalue\n", eig)
    eig_vr =eig_vr * (-1)
    print("eigen_v1\n",eig_vr[:,0])
    print("eigen_v2\n",eig_vr[:,1])
    print("eigen_v3\n",eig_vr[:,2])
    print("eigen_v4\n",eig_vr[:,3])
    delta_a = eig_vr[:,0] * ds.delta
    delta_b = eig_vr[:,1] * ds.delta
    delta_c = eig_vr[:,2] * ds.delta
    delta_d = eig_vr[:,3] * ds.delta

    delta_e = eig_vr[:,2] * 10 * ds.delta
    delta_f = eig_vr[:,2] * 10 * ds.delta
    delta_g = eig_vr[:,3] * 10 * ds.delta
    delta_h = eig_vr[:,3] * 10 * ds.delta
    ds.x0 = ds.x0[0,:]
    x_a = ds.x0 + delta_a
    x_b = ds.x0 + delta_b
    x_c = ds.x0 + delta_c
    x_d = ds.x0 + delta_d
    x_e = ds.x0 + delta_e
    x_f = ds.x0 + delta_f
    x_g = ds.x0 + delta_g
    x_h = ds.x0 + delta_h
    print("x_a",x_a)
    print("x_b",x_b)
    print("x_c",x_c)
    print("x_d",x_d)
    ds.x_a = x_a
    ds.x_b = x_b
    ds.x_c = x_c
    ds.x_d = x_d
    ds.x_e = x_e
    ds.x_f = x_f
    ds.x_g = x_g
    ds.x_h = x_h
def Eq_check(ds):
    eq = ds_func.equilibrium(ds)
    vp = eq[1,:].T
    #print("eq=",vp)
    print("eq=",eq[1,:].T)

    # for i in range(ds.xdim):
    #     F = ds.F.subs([(ds.sym_x, vp[0,:].T), (ds.sym_p, ds.params)])
    #     print(F'eq{i} = {F}')

    for i in range(ds.iter_max):
        F = ds.F.subs([(ds.sym_x, vp), (ds.sym_p, ds.params)])
        J = ds.F.jacobian(ds.sym_x)
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




def Condition(ds):
    F = sp.Matrix([
        # (dfdx(x0,lambda0) - mu_alpha I)(x_alpha - x0)
        (ds.dFdx.subs([(ds.sym_x, ds.x0.T), (ds.sym_p, ds.params)]) - ds.mu_alpha * np.eye(ds.xdim))
        @ (ds.x_alpha - ds.x0).T,
        # (dfdx(x0,lambda0) - mu_omegaI)(x_omega- x0)
        (ds.dFdx.subs([(ds.sym_x, ds.x0.T), (ds.sym_p, ds.params)]) - ds.mu_omega * np.eye(ds.xdim))
        @ (ds.x_omega - ds.x0).T,
        # (x_alpha - x0)(x_alpha - x0)^T - delta^2 = 0
        (ds.x_alpha - ds.x0) @ (ds.x_alpha - ds.x0).T - ds.delta * ds.delta ,
        # (x_omega- x0)(x_omega - x0)^T - delta^2 = 0
        (ds.x_omega - ds.x0) @ (ds.x_omega - ds.x0).T - ds.delta * ds.delta
        # phi(xalpha,lambda0, tau) - phi(xomega, lambda0,-tau) = 0

    ])
    return F
def jac(ds):
    F = Condition(ds)
    J = F.jacobian(ds.sym_x)
    return J



# def eigtest(ds):
#     a = np.array([[3,3],
#                   [5,1]])
#     print(a)
#     print(a.shape)
#     eig_test = linalg.eigvals(a)
#     print("固有値テスト", eig_test)
#     eig_vr = linalg.eig(a,left = False,right = True)[1]
#     print("右固有ベクトル\n",eig_vr)
#     eig_vl = linalg.eig(a,left = True,right = False)[1]
#     print("左固有ベクトル\n",eig_vl)
#     print(eig_vr[:,0])
#     dot = eig_vr[:,0] @ eig_vl[:,1]
#     print("内積\n", dot)

    

if __name__ == '__main__':
    main()