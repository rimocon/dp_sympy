import numpy as np
import sympy as sp

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

def eigen(x0, ds):

    # 固有値用の行列を定義
    eig = np.zeros([4,4],dtype = complex)

    for i in range(ds.xdim):
        # 平衡点の値をヤコビ行列に代入: df/dx(x0)してndarrayに変換
        np_jac = sp.matrix2numpy(ds.dFdx.subs([(ds.sym_x, x0.T.col(i)), (ds.sym_p, ds.params)]), dtype = np.float64)
        # 固有値,固有ベクトルを求める
        eig_temp, eigv_temp = np.linalg.eig(np_jac)
        # print(eigv_temp)
        eig[i,:] = eig_temp
    return eig
    