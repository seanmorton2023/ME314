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

    #print("KE of body 1:")
    ##display(KE_B1)

    #print("KE of body 2:")
    ##display(KE_B2)
    #print("Lagrangian:")
    ##display(lagrangian)
    #print("Simplified:")
    ##display(lagrangian_disp)

    #print("Euler-Lagrange equations:")
    ##display(total_eq)
    #print("Variables to solve for (transposed):")
    ##display(qdd.T)
    #display(total_eq_simpl)
    #display(total_eq_rounded)

    #for eq in eqns_solved:
    #    #display(eq)
    #for eq in eqns_new:
    #    #display(eq)

    #pickle the output of this constrained Euler-Lagrange derivation
    pass

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

###

def full_simulation(ICs):
    '''
    Carries out the simulate() function and saves results
    to a file. Encapsulating it in a file to get it out of
    my main() function.
    '''
    pkl_filename = '../dill/EL_simplified.dill'
    eqns_new = dill_load(pkl_filename)
    #print(eqns_new)

    #q_ext = sym.Matrix([q, q.diff(t), F_mat])

    #simulate the system with no impacts applied
    q_array = simulate(dxdt, ICs, t_span, dt, rk4)
    q_array = q_array.T

    #save q_array so we can animate it
    filename = '../csv/q_array.csv'
    write_csv_mat(filename, q_array) #from helpers.py

###

def ham_f():
    '''Generate the Hamiltonian function for this system.
    Only needs to be done once, not recalculated at every entry in
    the state array.
    '''
    #inertial properties defined in geometry.py
    KE_B1 = 0.5 * (VbB1.T @ inertia_B1 @ VbB1)[0]
    KE_B2 = 0.5 * (VbB2.T @ inertia_B2 @ VbB2)[0]
    U = m*g*(ym1 + ym2)
    ham_sym = KE_B1 + KE_B2 + U

    #sub in parameters from geometry.py
    print("Simplifying Hamiltonian...")
    ham_sym = ham_sym.simplify()
    ham_sym = ham_sym.subs(subs_dict)

    print("\nWriting to file:")
    ham_file = '../dill/hamiltonian.dill'
    dill_dump(ham_file, ham_sym)
    print(f"Wrote Hamiltonian to {ham_file}.")
    return ham_sym
 
def plot_results():
    '''For reading Hamiltonian + trajectory results from files
    and plotting them over time.
    '''
     #plot Hamiltonian array over time
    q_array = pd.read_csv('../csv/q_array.csv', header=None).to_numpy()
    ham_sym = dill_load('../dill/hamiltonian.dill')
    q_noforces = sym.Matrix([q, q.diff(t)])
    ham_np = sym.lambdify(q_noforces, ham_sym)
    ham_array = [ham_np(*s) for s in q_array]

    plt.figure(1)
    plt.plot(ham_array)
    plt.xlabel("Index")
    plt.ylabel("Hamiltonian value (J)")
    plt.title("Hamiltonian for current simulation")

    #plot trajectory over time
    plt.figure(2)
    plot_array = q_array.T
    t_array = np.linspace(t_span[0],t_span[1],len(plot_array[0]))
    legend = ['x','y','theta1','theta2','phi1','phi2']

    for i in range(6):
        state = plot_array[i]
        plt.plot(t_array, state, label=legend[i])
        
    plt.xlabel("Index")
    plt.ylabel("State variable value")
    plt.title("Trajectory for current simulation")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    #F_mat = sym.Matrix([
    #    sym.symbols(r'F_x'),
    #    sym.symbols(r'F_y'),
    #    sym.symbols(r'F_\theta1'),
    #    sym.symbols(r'F_\theta2'),
    #    sym.symbols(r'F_\phi1'),
    #    sym.symbols(r'F_\phi2'),
    #])

    #eqns_new = dill_load('../dill/EL_simplified.dill')
    #q_ext = sym.Matrix([q, q.diff(t), F_mat])
    #xdd_np, ydd_np, theta1dd_np, theta2dd_np, phi1dd_np, phi2dd_np \
    #    = construct_dxdt(eqns_new, q_ext)
    #dxdt = construct_dxdt(eqns_new, q_ext)
    F_eqs_array = np.array([
        lambda t: 0, #F_x
        lambda t: 19.62, #F_y
        lambda t: 0, #F_theta1
        lambda t: 0, #F_theta2
        lambda t: 0, #F_phi1
        lambda t: 0, #F_phi2
    ])

    dxdt = construct_dxdt(F_eqs_array)


    ##for Lagrangian debug - save and then load into Jupyter NB
    #lagrangian = compute_lagrangian()
    #lagrangian_filename = '../dill/lagrangian.dill'
    #dill_dump(lagrangian_filename, lagrangian)

    # save output of Euler-Lagrange equations for later use/reuse
    #eqns_new = compute_solve_EL(F_mat)
    #temp = eqns_new
    #pkl_filename = '../dill/EL_simplified.dill'
    #dill_dump(pkl_filename, temp) 

    t_span = [0, 10]
    dt = 0.01

    theta0 = np.pi/4
    phi0 = np.pi/3
    init_posns = [1, 1, theta0, -theta0, phi0, -phi0]
    init_velocities = [0, 0, 0, 0, 0, 0]
    ICs = init_posns + init_velocities
    full_simulation(ICs) #saves results to q array

