import sys
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import sin,cos
import json
import dynamical_system
import ds_func

def main():
    # load data from json file
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)
    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    ds = dynamical_system.DynamicalSystem(json_data)

    eq = ds_func.equilibrium(ds)
    eig = ds_func.eigen(eq, ds)

if __name__ == '__main__':
    main()