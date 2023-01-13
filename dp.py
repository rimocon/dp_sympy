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


def func(x, p, c):
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

   

    tau = 10
    ds.x0 = Eq_check(ds.params,ds)
    Eigen(ds.x0,ds.params,ds)
    vp = np.block([ds.x_alpha,ds.x_omega,ds.p[ds.var],ds.duration[1]])
    print("vpshape",vp.shape)
    # ds.sym_F = ds_func.newton_F(ds.sym_z,ds)
    # print("sym_F= ",ds.sym_F)
    # ds.sym_J = ds.sym_F.jacobian(ds.sym_z)
    # print("sym_J= ",ds.sym_J)
    newton_method(vp,ds)
   

def Condition(z,ds):
    F = ds.sym_F.subs(ds.sym_z,z)
    J = ds.sym_J.subs(ds.sym_z,z)
    F = ds_func.sp2np(F)
    J = ds_func.sp2np(J)
    dFdlambda = J[:,8+ds.var].reshape(10,1)
    dFdtau = np.zeros((10,1))
    J = np.block([[J[:,0:8],dFdlambda,dFdtau]])
    # state_p = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    # state_m = -1 * state_p
    for i in range(0,4):
        # dphidlambda(plus) - dphidlambda(minus)
        J[i+6][8] = ds.state_p.y[20+i,-1] - ds.state_m.y[20+i,-1]
        J[i+6][9] = 1
        for j in range(0,4):
            # dphidxalpha ~ dphidxomega
            J[i+6][j] = ds.state_p.y[4+i+4*j,-1]
            J[i+6][j+4] = ds.state_m.y[4+i+4*j,-1]
    return F,J


def Eigen(x0,p,ds):
    #パラメータに依存するように
    # eq = ds_func.equilibrium(ds)
    eq = ds_func.set_x0(p,ds.c) 
    x0 = ds_func.sp2np(eq)
    x0 = x0[0,:]
    print("x0\n",x0)
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

def Eq_check(p,ds):
    eq = ds_func.set_x0(p,ds.c)
    vp = eq[0,:].T
    vp = sp.Matrix(vp)
    print("eq=",vp)

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
    print("ppppppp",p)
    print("vp",vp)
    ds.x0_p = np.concatenate([vp[0:4],ds.vari_ini])
    ds.x0_m = np.concatenate([vp[4:8],ds.vari_ini])
    print("initial value",ds.x0_p)

    # for plus
    ds.state_p = solve_ivp(variation.func, [0,vp[9]], ds.x0_p,
        method='RK45', args = (p, c),
        rtol=1e-12)
    print("y_p",ds.state_p.y)
    print("y0 - y4",ds.state_p.y[0:4,-1])
    ## for minus
    ds.state_m = solve_ivp(variation.func, [-vp[9],0], ds.x0_m,
        method='RK45', args = (p, c), 
        rtol=1e-12)
    # # print("t_m",ds.state_m.t)
    print("y_m",ds.state_m.y)
    a = ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
    print("test",a)

# ニュートン法
def newton_method(vp,ds):
    p = ds.p
    for i in range(ds.ite_max): 
        # パラメタだけセットsetto
        p[ds.var] = vp[8]
        # 微分方程式+変分方程式を初期値vpで解く   
        solve(vp,ds)
        z = np.block([vp[0:4],vp[4:8],p])
        z = sp.Matrix(z.reshape(12,1))
        #######ここ1回だけでいいからどうにかしたい・・・
        ds.sym_F = ds_func.newton_F(ds.sym_z,ds)
        print("sym_F= ",ds.sym_F)
        ds.sym_J = ds.sym_F.jacobian(ds.sym_z)
        print("sym_J= ",ds.sym_J)
        ################################3
        F,J = Condition(z,ds)
        print("F",F)
        print("Fshape",F.shape)
        print("J",J)
        print("J shape",J)
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
        print("test=",test)
        vn = np.linalg.solve(J,-F).flatten() + vp
        print("vn=",vn)
        vp = vn
        

if __name__ == '__main__':
    main()