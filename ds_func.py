from re import A
import numpy as np
import sympy as sp
from scipy import linalg

def equilibrium(ds):
    t1 = sp.acos((ds.params[2] - ds.params[3]) / ((ds.const[0] + ds.const[1]) * ds.const[2] * ds.const[4]))
    t2 = sp.acos(ds.params[3] / (ds.const[1] * ds.const[3] * ds.const[4])) 

    eqpoints = sp.Matrix([
        [t1, 0, t2 - t1, 0],
        [-t1, 0, t2 + t1, 0],
        [t1, 0, -t2 - t1, 0],
        [-t1, 0, -t2 + t1, 0]
    ])
    return eqpoints

def eigen(x0, ds, n):

    # subs eqpoints for jacobian(df/dx(x0)) 
    # jac = ds.dFdx.subs([(ds.sym_x, x0), (ds.sym_p, ds.params)])
    # for all eqpoints
    jac = ds.dFdx.subs([(ds.sym_x, x0.T.col(n)), (ds.sym_p, ds.params)])
    # convert to numpy
    np_jac = sp2np(jac)
    # calculate eigen values,eigen vectors
    eig_vals,eig_vl, eig_vr = linalg.eig(np_jac, left=True,right=True)
    
    '''
    ####### for all eqpoints  #######
    for i in range(ds.xdim):
        # subs eqpoints for jacobian(df/dx(x0)) 
        jac = ds.dFdx.subs([(ds.sym_x, x0.T.col(i)), (ds.sym_p, ds.params)])
        # convert to numpy
        np_jac = sp2np(jac)
        # calculate eigen values,eigen vectors
        eig_temp,eig_vl, eig_vr = linalg.eig(np_jac)
        # print(eigv_temp)
        eig[i,:] = eig_temp
    '''
    return eig_vals,eig_vl,eig_vr
    
def sp2np(x):
    return sp.matrix2numpy(x, dtype = np.float64)


def dFdx(x, ds):
    dfdx = ds.dFdx.subs([(ds.sym_x, x), (ds.sym_p, ds.params)])
    return sp2np(dfdx)


def Condition(z,ds):
    dfdx = ds.dFdx.subs([(ds.sym_x, ds.x0), (ds.sym_p, ds.params)])
    print(dfdx)
    I = sp.eye(4)
    x0 = sp.Matrix(ds.x0).reshape(4,1)
    x_alpha = sp.Matrix([z[0],z[1],z[2],z[3]])
    x_omega = sp.Matrix([z[4],z[5],z[6],z[7]])
    print("x_alpha",x_alpha)
    print("x_omega",x_omega)
    print("mu_alpha",ds.mu_alpha)
    print("mu_alpha",ds.mu_omega)
    sym_F = sp.Matrix([
        # 固有ベクトル上にx_alphaが存在する条件(dfdx(x0,lambda) - mu_alpha I)@(x_alpha - x0)
        (dfdx - ds.mu_alpha * I) @ (x_alpha - x0),
        ((x_alpha - x0).T @ (x_alpha - x0))[0,0] - ds.delta * ds.delta,
        (dfdx - ds.mu_omega * I) @ (x_omega- x0),
        ((x_omega- x0).T @ (x_omega- x0))[0,0] - ds.delta * ds.delta
    ])
    
    f = sp2np(sym_F)
    print(f)
    print(sym_F)

def newton_F(z, ds):

    dfdx = ds.dFdx_n.subs([(ds.sym_x, ds.x0)])
    print("dfdx",dfdx)
    I = sp.eye(4)
    x0 = sp.Matrix(ds.x0).reshape(4,1)
    x_alpha = sp.Matrix([z[0],z[1],z[2],z[3]])
    x_omega = sp.Matrix([z[4],z[5],z[6],z[7]])
    lamb = sp.Matrix([z[8],z[9],z[10],z[11]])
    
    ret = sp.Matrix([
        # 固有ベクトル上にx_alphaが存在する条件(dfdx(x0,lambda) - mu_alpha I)@(x_alpha - x0)
        ((dfdx - ds.mu_alpha * I) @ (x_alpha - x0))[0:3,0],
        # deltaだけ離れた点である条件
        ((x_alpha - x0).T @ (x_alpha - x0))[0,0] - ds.delta * ds.delta,
        # x_omegaも同様
        ((dfdx - ds.mu_omega * I) @ (x_omega- x0))[0:3,0],
        ((x_omega- x0).T @ (x_omega- x0))[0,0] - ds.delta * ds.delta,
        # 解が一致する条件
        ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
        
    ])
    return ret