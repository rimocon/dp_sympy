import numpy as np
import sympy as sp
from scipy import linalg

def equilibrium(ds):
    theta_1 = sp.acos((ds.params[2] - ds.params[3]) / ((ds.const[0] + ds.const[1]) * ds.const[2] * ds.const[4]))
    theta_2 = sp.acos(ds.params[3] / (ds.const[1] * ds.const[3] * ds.const[4])) - theta_1

    eqpoints = sp.Matrix([
        [theta_1, 0, -theta_2, 0],
        [-theta_1, 0, -2 * theta_1 + theta_2, 0],
        [theta_1, 0, 2 * theta_1 - theta_2, 0],
        [-theta_1, 0, theta_2, 0]
    ])
    return eqpoints

def eigen(x0, ds, n):

    # subs eqpoints for jacobian(df/dx(x0)) 
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
