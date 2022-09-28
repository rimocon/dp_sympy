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

    eq = ds_func.equilibrium(ds)
    ds.x0 = ds_func.sp2np(eq)
    print("x0\n",ds.x0)
    '''
    for i in range(4):
        eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)
        ds.mu_alpha = eig[0]
        ds.mu_omega = eig[1]

        print("eigenvalue\n", eig)
    '''
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
    

    '''
    ##### for all eqpoints#####
    for i in range(ds.xdim):
        delta_vec = eig_vr[:,i] * ds.delta
        x_alpha = np_eq + delta_vec
    '''
    print("x_alpha\n",x_alpha)
    print("x_omega\n",x_omega)

    test = Condition(ds)
    # print("条件式テスト",test)
    jac_test = jac(ds)
    # print("ヤコビアンテスト",jac_test)


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