from re import A
import sys
import json
import dynamical_system
import ds_func
# import variation
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import sin,cos
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
    # 初期値に関する変分方程式
    dFdx = np.array([[0, 1, 0, 0],
    # 2行目
    [(9.80665*(-1.0*cos(x[2]) - 1.0)*sin(x[0] + x[2]) + 9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0) ,
    (-2.0*(-1.0*cos(x[2]) - 1.0)*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3] - 1.0*p[0])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    0.111111111111111*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))*(-(1.0*cos(x[2]) + 1.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]) + 2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - 1.0*p[0]*x[1] + 1.0*p[2])/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(-1.0*cos(x[2]) - 1.0) + 1.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-(-1.0*cos(x[2]) - 1.0)*p[1] + 2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0, 0, 1],
    # ４行目
    [((9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))*(-1.0*cos(x[2]) - 1.0) + 9.80665*(2.0*cos(x[2]) + 3.0)*sin(x[0] + x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[3] - p[0])*(-1.0*cos(x[2]) - 1.0) - 2.0*(2.0*cos(x[2]) + 3.0)*sin(x[2])*x[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    0.111111111111111*(-(1.0*cos(x[2]) + 1.0)*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2]) + (2.0*cos(x[2]) + 3.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]))*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(2.0*cos(x[2]) + 3.0) + (-1.0*cos(x[2]) - 1.0)*(9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2) - 2.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 1.0*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2])*sin(x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])*(-1.0*cos(x[2]) - 1.0) - (2.0*cos(x[2]) + 3.0)*p[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])

    dphidx = np.array(x[4:24])
    ## dFdx @ dphidx0 (要はtheta_10に関するやつ)
    i_0 = (dFdx @ dphidx.reshape(20,1)[0:4]).reshape(4,)
    ## dFdx @ dphidx1 (omega_10に関するやつ)
    i_1 = (dFdx @ dphidx.reshape(20,1)[4:8]).reshape(4,)
    ## dFdx @ dphidx2
    i_2 = (dFdx @ dphidx.reshape(20,1)[8:12]).reshape(4,)
    ## dFdx @ dphidx3
    i_3 = (dFdx @ dphidx.reshape(20,1)[12:16]).reshape(4,)

    # ## パラメタに関する変分方程式
    dFdl = np.array([[0, 0, 0, 0],
    # 2行目
    [-1.0*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(-1.0*cos(x[2]) - 1.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    1.0/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0 ,0, 0],
    # 4行目
    [-(-1.0*cos(x[2]) - 1.0)*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(2.0*cos(x[2]) + 3.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (2.0*cos(x[2]) + 3.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])
    ##この4本から3本だけでいい
    # ここの0:4,3←ここは選択するパラメタによって変える
    p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdl[0:4,3].reshape(4,1)).reshape(4,)
    ## 元の微分方程式+変分方程式に結合
    ret = np.concatenate([ret,i_0,i_1,i_2,i_3,p])
    # print("ret",ret)
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

   

    # tau = 10
    ds.x0 = Eq_check(ds.params,ds)
    Eigen(ds.x0,ds.params,ds)
    vp = np.block([ds.x_alpha,ds.x_omega,ds.p[ds.var],ds.duration[1]])
    # print("vpshape",vp.shape)
    solve(vp,ds)
    ds.sym_F = ds_func.newton_F(ds.sym_z,ds)
    # print("sym_F= ",ds.sym_F)
    ds.sym_J = ds.sym_F.jacobian(ds.sym_z)
    # print("sym_J= ",ds.sym_J)
    newton_method(vp,ds)
   

def Condition(z,ds):
    F = ds.sym_F.subs([(ds.sym_x,ds.x0),(ds.sym_z,z)])
    J = ds.sym_J.subs([(ds.sym_x,ds.x0),(ds.sym_z,z)])
    F = ds_func.sp2np(F)
    J = ds_func.sp2np(J)
    dFdlambda = J[:,8+ds.var].reshape(10,1)
    dFdtau = np.zeros((10,1))
    J = np.block([[J[:,0:8],dFdlambda,dFdtau]])
    for i in range(0,4):
        # dphidlambda(plus) - dphidlambda(minus)
        J[i+6][8] = ds.state_p.y[20+i,-1] - ds.state_m.y[20+i,-1]
        for j in range(0,4):
            # dphidxalpha ~ dphidxomega
            J[i+6][j] = ds.state_p.y[4+i+4*j,-1]
            J[i+6][j+4] = ds.state_m.y[4+i+4*j,-1]
    p = ds.p
    p[ds.var] = z[8+ds.var]
    M1 = ds.c[0]
    M2 = ds.c[1]
    L1 = ds.c[2]
    L2 = ds.c[3]
    G = ds.c[4]
    x = []
    x[0:4] = ds.state_p.y[0:4,-1]
    a11p= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12p = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21p = a12p
    a22p = M2 * L2 * L2
    b1p = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2p = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    deltap = a11p * a22p - a12p * a21p

    x[0:4] = ds.state_m.y[0:4,-1]
    a11m= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12m = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21m = a12p
    a22m = M2 * L2 * L2
    b1m = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2m = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    deltam = a11p * a22p - a12p * a21p
    J[6][9] = ds.state_p.y[1,-1] - ds.state_m.y[1,-1]
    J[7][9] = (b1p * a22p - b2p * a12p) / deltap - (b1m * a22m - b2m * a12m) / deltam
    J[8][9] = ds.state_p.y[3,-1] - ds.state_m.y[3,-1]
    J[9][9] = (b2p * a11p - b1p * a21p) / deltap - (b2m * a11m - b1m * a21m) / deltam
    # print("J[6][9]",J[6][9])
    # print("J[7][9]",J[7][9])
    # print("J[8][9]",J[8][9])
    # print("J[9][9]",J[9][9])
    return F,J


def Eigen(x0,p,ds):
    #パラメータに依存するように
    # eq = ds_func.equilibrium(ds)
    eq = ds_func.set_x0(p,ds.c) 
    x0 = ds_func.sp2np(eq)
    x0 = x0[0,:]
    # print("x0\n",x0)
    # for i in range(4):
    #     eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, i)

    eig,eig_vl,eig_vr = ds_func.eigen(eq, ds, 0)
    print("eigenvalue\n", eig)
    ds.mu_alpha = eig[1].real
    ds.mu_omega = eig[2].real
    eig_vr = eig_vr * (-1)
    print("eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xa = x0 + delta

    eig_vr = eig_vr * (-1)
    # print("inverse_eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    ds.xb = x0 + delta

    ####ここは８本から２本選択
    ds.x_alpha = ds.xa[1,:]
    ds.x_omega = ds.xa[2,:]
    print("x0", x0)
    print("x_alpha", ds.x_alpha)
    print("x_omega", ds.x_omega)

def Eq_check(p,ds):
    eq = ds_func.set_x0(p,ds.c)
    vp = eq[0,:].T
    vp = sp.Matrix(vp)
    # print("eq=",vp)

    # for i in range(ds.xdim):
    #     F = ds.F.subs([(ds.sym_x, vp[0,:].T), (ds.sym_p, ds.params)])
    #     print(F'eq{i} = {F}')

    J = ds.F.jacobian(ds.sym_x)
    for i in range(ds.iter_max):
        F = ds.F.subs([(ds.sym_x, vp), (ds.sym_p, p)])
        J = J.subs([(ds.sym_x, vp), (ds.sym_p, p)])
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
        test = np.linalg.solve(J,-F)
        print(test)
        vn = np.linalg.solve(J,-F) + vp
        print("i=",i)
        print("vn=",vn)
        vp = vn
        # if vn[5] > 1.0:
        #   print("B0 is too high")




def solve(vp,ds):
    ####solve####
    # import numpy parameter
    # パラメータ
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    p[ds.var] = vp[8]
    # print("ppppppp",p)
    # print("vp",vp)
    ds.x0_p = np.concatenate([vp[0:4],ds.vari_ini])
    ds.x0_m = np.concatenate([vp[4:8],ds.vari_ini])
    # print("initial value",ds.x0_p)

    # for plus
    ds.state_p = solve_ivp(func, [0,vp[9]], ds.x0_p,
        method='RK45', args = (p, c),
        rtol=1e-12)
    # print("y_p",ds.state_p.y)
    ## for minus
    ds.state_m = solve_ivp(func, [-vp[9],0], ds.x0_m,
        method='RK45', args = (p, c), 
        rtol=1e-12)
    # print("t_m",ds.state_m.t)
    # print("y_m",ds.state_m.y)

# ニュートン法
def newton_method(vp,ds):
    p = ds.p
    for i in range(ds.ite_max):
        print(f"###################iteration:{i}#######################")
        # パラメタだけセット
        p[ds.var] = vp[8]
        # パラメタによって平衡点は変化するのでセットしなおし
        ds.x0 = (ds_func.set_x0(p,ds.c)[0,:]).reshape(4,1)
        # print("x0",ds.x0)
        # 微分方程式+変分方程式を初期値vpで解く
        solve(vp,ds)
        z = np.block([vp[0:4],vp[4:8],p])
        # print("z=",z)
        z = sp.Matrix(z.reshape(12,1))
        ################################3
        F,J = Condition(z,ds)
        print("F",F)
        print("J",J)
        dif = abs(np.linalg.norm(F))
        print("diff=",dif)
        if dif < ds.eps:
            print("success!!!")
            print("solve vp = ",vp)
            return vp
        if dif > ds.explode:
            print("Exploded")
            exit()
        test = np.linalg.solve(J,-F)
        print("solve(J,-F) = ",test)
        vn = np.linalg.solve(J,-F).flatten() + vp
        print("vn=",vn)
        vp = vn
        

if __name__ == '__main__':
    main()