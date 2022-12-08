import numpy as np
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt

import dill
from tqdm import tqdm

from geometry import *
from helpers import *

#from IPython.core.display import display

def rk4(dxdt, x, t, dt):
    '''
    Applies the Runge-Kutta method, 4th order, to a sample function,
    for a given state q0, for a given step size. Currently only
    configured for a 2-variable dependent system (x,y).
    ==========
    dxdt: a Sympy function that specifies the derivative of the system of interest
    t: the current timestep of the simulation
    x: current value of the state vector
    dt: the amount to increment by for Runge-Kutta
    ======
    returns:
    x_new: value of the state vector at the next timestep
    '''      
    k1 = dt * dxdt(t, x)
    k2 = dt * dxdt(t + dt/2.0, x + k1/2.0)
    k3 = dt * dxdt(t + dt/2.0, x + k2/2.0)
    k4 = dt * dxdt(t + dt, x + k3)
    x_new = x + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    
    return x_new

def simulate(f, x0, tspan, dt, integrate):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    
    Parameters
    ============
    f: Python function
        derivate of the system at a given step x(t), 
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        initial conditions
    tspan: Python list
        tspan = [min_time, max_time], it defines the start and end
        time of simulation
    dt:
        time step for numerical integration
    integrate: Python function
        numerical integration method used in this simulation

    Return
    ============
    x_traj:
        simulated trajectory of x(t) from t=0 to tf
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
       
    print("\nSimulating:")
    for i in tqdm(range(N)):
        t = tvec[i]
        xtraj[:,i]=integrate(f,x,t,dt)
        x = np.copy(xtraj[:,i])
    return xtraj 

def compute_lagrangian():
    
    #define kinetic and potential energy
    KE_B1 = 0.5 * (VbB1.T @ inertia_B1 @ VbB1)[0]
    KE_B2 = 0.5 * (VbB2.T @ inertia_B2 @ VbB2)[0]
    U = m*g*(ym1 + ym2)

    lagrangian = KE_B1 + KE_B2 - U
    return lagrangian

def compute_solve_EL(F_mat):
    '''
    Encapsulate the solving process in a function for the sake of 
    not having to run this again every time. For the sake of reducing
    stack calls, could put this in its own file instead.
    '''

    #calculate forced Euler-Lagrange equations. no forces of constraint
    qd = q.diff(t)
    qdd = qd.diff(t)

    lagrangian = compute_lagrangian()
    lhs = compute_EL_lhs(lagrangian, q, t)
    RHS = sym.zeros(len(lhs), 1)
    RHS = RHS + F_mat

    lhs = lhs.subs(subs_dict) #defined in geometry.py
    total_eq = sym.Eq(lhs, RHS)

    #do symbolic substitutions before solving to speed up computation
    print("\nPress any key to simplify the E-L equations. ", end='')
    input()
    print("Simplifying:")

    #waited on all simplify() calls until here
    t0 = time.time()
    total_eq_simpl = total_eq.simplify()
    tf = time.time()

    print(f"\ntotal_eq.simplify(): \nElapsed: {round(tf - t0, 2)} seconds")

    #attempt to round near-zero values to zero. source: https://tinyurl.com/f7t8wbmw
    total_eq_rounded = total_eq_simpl
    for a in sym.preorder_traversal(total_eq_simpl):
        if isinstance(a, sym.Float):
            total_eq_rounded = total_eq_rounded.subs(a, round(a, 8))
        
    print("Euler-Lagrange equations - simplified:")
    #display(total_eq_simpl)
    print("Euler-Lagrange equations - rounded:")
    #display(total_eq_rounded)

    print("\nPress any key to solve the E-L equations. ", end='')
    input()
    print("Solving:")

    t0 = time.time()
    #soln = sym.solve(total_eq, qdd, dict = True, simplify = False, manual = True)
    #soln = sym.solve(total_eq, qdd, dict = True, manual = True)
    soln = sym.solve(total_eq_rounded, qdd, dict = True, simplify = False)

    tf = time.time()
    print(f"\nsym.solve(): \nElapsed: {round(tf - t0, 2)} seconds")

    eqns_solved = format_solns(soln)

    #simplify equations one by one
    eqns_new = []
    print("\nSimplifying EL equations:")
    for eq in tqdm(eqns_solved):
        eq_new = sym.simplify(eq)
        eqns_new.append(eq_new)

    return eqns_new

def construct_dxdt(f_eqs_array):
    '''Generates our dynamics function dxdt() using the
    second-derivative equations derived from the Euler-Lagrange
    equations.

    Arguments:
    - f_eqs_array: an array of Numpy functions, lambda functions, or
        other univatiate functions of time

    Returns: dxdt, a function f(t,s)
    '''
    F_mat = sym.Matrix([
        sym.symbols(r'F_x'),
        sym.symbols(r'F_y'),
        sym.symbols(r'F_\theta1'),
        sym.symbols(r'F_\theta2'),
        sym.symbols(r'F_\phi1'),
        sym.symbols(r'F_\phi2'),
    ])

    eqns_new = dill_load('../dill/EL_simplified.dill')
    q_ext = sym.Matrix([q, q.diff(t), F_mat])

    #lambdify the second derivative equations and construct dynamics function
    xdd_sy      = eqns_new[0].rhs
    ydd_sy      = eqns_new[1].rhs
    theta1dd_sy = eqns_new[2].rhs
    theta2dd_sy = eqns_new[3].rhs
    phi1dd_sy   = eqns_new[4].rhs
    phi2dd_sy   = eqns_new[5].rhs

    xdd_np      = sym.lambdify(q_ext,      xdd_sy)
    ydd_np      = sym.lambdify(q_ext,      ydd_sy)
    theta1dd_np = sym.lambdify(q_ext, theta1dd_sy)
    theta2dd_np = sym.lambdify(q_ext, theta2dd_sy)
    phi1dd_np   = sym.lambdify(q_ext,   phi1dd_sy)
    phi2dd_np   = sym.lambdify(q_ext,   phi2dd_sy)


    def dxdt(t,s):

        F_array = [f(s,t) for f in f_eqs_array]
        s_ext = np.append(s, F_array)
        #format of s_ext: 
        #0-5: state values
        #6-11: values of derivative of state
        #12-17: values of force at given time
    
        return np.array([
            *s[6:12],         
            xdd_np(*s_ext),      
            ydd_np(*s_ext),      
            theta1dd_np(*s_ext), 
            theta2dd_np(*s_ext), 
            phi1dd_np(*s_ext),   
            phi2dd_np(*s_ext),
        ])

    #return type is a function
    return dxdt
