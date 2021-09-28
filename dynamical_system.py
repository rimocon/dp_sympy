import numpy as np
import sympy as sp
from sympy import sin, cos

def map(x, p, c):
    # double pendulum

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

    ret = sp.Matrix([x[1],
                    (b1 * a22 - b2 * a12) / delta,
                    x[3],
                    (b2 * a11 - b1 * a21) / delta
                    ])
    return ret


class DynamicalSystem:
    x0 = []
    params = []
    const = []
    xdim = 0
    
    sym_x = 0
    sym_p = 0
    dFdx = []

    def __init__(self, json):
        self.x0 = sp.Matrix(json['x0'])
        self.params = sp.Matrix(json['params'])
        self.const = sp.Matrix([json['const']])
        self.xdim = json['xdim']

        self.sym_x = sp.MatrixSymbol('x', self.xdim, 1)
        self.sym_p = sp.MatrixSymbol('p', sp.shape(self.params)[0],1)
        self.F = map(self.sym_x, self.sym_p, self.const)
        self.dFdx = self.F.jacobian(self.sym_x)
        
        