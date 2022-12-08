import numpy as np
import sympy as sym
import pandas as pd

import dill
import time
from datetime import datetime
from tqdm import tqdm

from geometry import *
from helpers import *
from el_equations import *

def calculate_sym_vertices():
    '''
    Calculates 16 symbolic expressions to describe the vertices of the
    2 boxes in the system - 4 vertices * 2 coords(x,y) * 2 boxes.
    
    This is a moderately time-consuming operation (3min) so the output
    will be saved to a file to prevent losing data.

    Returns: 16 symbolic expressions for vijn_Bk, where i = 1-2 (box#), 
    j = 1-4 (vertex#), k = 2-1 (opposite of i #)
    '''

    #define positions of vertices in boxes' home frames
    v1bar = sym.Matrix([ w/2,  w/2, 0, 1])
    v2bar = sym.Matrix([-w/2,  w/2, 0, 1])
    v3bar = sym.Matrix([-w/2, -w/2, 0, 1])
    v4bar = sym.Matrix([ w/2, -w/2, 0, 1])

    #from geometry.py
    GB1B2 = InvSEn(GsB1) @ GsB2
    GB2B1 = InvSEn(GB1B2)

    vbar_list = [v1bar, v2bar, v3bar, v4bar]
    g_list = [GB2B1, GB1B2]

    #do this algorithmically so we can wrap it in a tqdm; track progress
    vertices_coords_list = []
    print("Calculate_sym_vertices(): simplifying vertex coords.")
    for i in tqdm(range(8)):
        #G: 00001111, vbar: 01230123
        vij_Bk = sym.simplify(g_list[i//4] @ vbar_list[i%4])
        vertices_coords_list.append(vij_Bk)

    #save results
    print('\nSaving results:')
    filepath = '../dill/vertices_coords_list.dill'
    dill_dump(filepath, vertices_coords_list)
    print(f"Vertices coords saved to {filepath}.")

def convert_coords_to_xy():
    '''Take the symbolic coordinates we found for the two boxes and 
    split them into x and y components.
    '''
    vertices_coords_list = dill_load('../dill/vertices_coords_list.dill')
    vertices_xy_list = []
    for coord in vertices_coords_list:
        coordx, coordy, _, _ = coord
        vertices_xy_list.append([coordx, coordy])
        pass

    #flatten
    vertices_list_sym = np.array(vertices_xy_list).flatten().tolist()
    return vertices_list_sym

def calculate_sym_phiq(vlist):
    '''
    Calculate the symbolic impact equations Phi(q) for use in 
    applying impact updates to the system. There will be 32 
    phi(q) equations - 2 per possible vertex + side of impact
    combination.

    Saves symbolic phi(q) to a pickled file for loading
    and use in other files, plus saving variables between sessions
    of Python/Jupyter notebook.

    Arguments:
    vlist - a list of symbolic vertex coordinates broken apart 
        into x and y

    Returns: None; load symbolic phi(q) from file for simplicity
    '''

    phiq_list = []
    for vertex in vlist:
        phiq_list.append(vertex + w/2) #impact from left side
        phiq_list.append(vertex - w/2) #impact from right side

    #substitute in values of L and w
    phiq_list = [expr.subs(subs_dict) for expr in phiq_list]
    return phiq_list

def impact_condition(s):
    '''Contains and evaluates an array of impact conditions for the current
    system, at the current state s. 
    
    Returns: a logical true/false as to whether any impact condition was met;
        list of indices of impact conditions that were met
    '''

    v11x_B2_np, v11y_B2_np, v12x_B2_np, v12y_B2_np, \
    v13x_B2_np, v13y_B2_np, v14x_B2_np, v14y_B2_np, \
    v21x_B1_np, v21y_B1_np, v22x_B1_np, v22y_B1_np, \
    v23x_B1_np, v23y_B1_np, v24x_B1_np, v24y_B1_np  = vertices_list_np
    
    #define tolerance for impact condition
    ctol = 1/24.0 #proportional to w/2
    bound = w_num/2.0 + ctol

    impact_conds = np.array([
        -bound < v11x_B2_np(*s) < bound   and   -bound < v11y_B2_np(*s) < bound,
        -bound < v12x_B2_np(*s) < bound   and   -bound < v12y_B2_np(*s) < bound,
        -bound < v13x_B2_np(*s) < bound   and   -bound < v13y_B2_np(*s) < bound,
        -bound < v14x_B2_np(*s) < bound   and   -bound < v14y_B2_np(*s) < bound,
        
        -bound < v21x_B1_np(*s) < bound   and   -bound < v21y_B1_np(*s) < bound,
        -bound < v22x_B1_np(*s) < bound   and   -bound < v22y_B1_np(*s) < bound,
        -bound < v23x_B1_np(*s) < bound   and   -bound < v23y_B1_np(*s) < bound,
        -bound < v24x_B1_np(*s) < bound   and   -bound < v24y_B1_np(*s) < bound,     
    ])
    
    #find any impact conditions that have been met
    impact_met = np.any(impact_conds)
    impact_indices = np.nonzero(impact_conds)[0].tolist() #indices where true
    
    #print("Impact condition debug:")

    #vertex_posn_values = np.array([
    #    v11x_B2_np(*s), v11y_B2_np(*s),
    #    v12x_B2_np(*s), v12y_B2_np(*s),
    #    v13x_B2_np(*s), v13y_B2_np(*s),
    #    v14x_B2_np(*s), v14y_B2_np(*s),
        
    #    v21x_B1_np(*s), v21y_B1_np(*s),
    #    v22x_B1_np(*s), v22y_B1_np(*s),
    #    v23x_B1_np(*s), v23y_B1_np(*s),
    #    v24x_B1_np(*s), v24y_B1_np(*s),     
    #])

    #print(f"Boundary for impact detection: {bound}")
    #for i in range(len(vertex_posn_values)//2):
    #    print(f"Vertex V{(i//4)+1}{(i%4)+1} x,y values: {round(vertex_posn_values[2*i],3)}  {round(vertex_posn_values[(2*i)+1],3)}")

    #if impact_met:
    #    #print("\nImpact condition debug:")
    #    ##print("V11x, V11y, V21x, and V21y values:")
    #    #print("V12x, V12y, V21x, and V21y values:")
    #    #print(v12x_B2_np(*s))
    #    #print(v12y_B2_np(*s))
    #    #print(v21x_B1_np(*s))
    #    #print(v21y_B1_np(*s))

    #    #print("\nV11x, V11y, V22x, and V22y values:")
    #    #print(v11x_B2_np(*s))
    #    #print(v11y_B2_np(*s))
    #    #print(v22x_B1_np(*s))
    #    #print(v22y_B1_np(*s))
    #    pass

    return impact_met, impact_indices

def phi_nearzero(s, atol):
    '''Takes in the current system state s, and returns the indices of
    impact conditions phi(q) that are close to 0 at the given instant in
    time. 
    
    The results of this function will then be used to determine which of
    the symbolically solved impact updates will be applied to the system.
    
    Parameters:
    - s: current state of system: x, y, theta1, theta2, phi1, phi2.
    - atol: absolute tolerance for "close to zero", passed to np.isclose().
        Example: atol = 1E-4
    
    Returns:
    - a list of indices of phi that are near zero at the given time.
    '''

    #apply upper and lower bound condition to all vertices, in x and y directions
    phi_arr_np = np.array([])
    for i in np.arange(0, len(vertices_list_np)):
        phi_arr_np = np.append(phi_arr_np, vertices_list_np[i](*s) + w_num/2.0)
        phi_arr_np = np.append(phi_arr_np, vertices_list_np[i](*s) - w_num/2.0)

    #we're interested in which of the phi conditions were evaluated at close to 0, so return the
    #list of indices close to 0
    closetozero = np.isclose( phi_arr_np, np.zeros(phi_arr_np.shape), atol=atol )
    
    #print("\nPhi_nearzero debug:")
    #print(f"\nPhi_arr_np: \n{phi_arr_np}")


    #this gives the indices of phi close to 0
    any_nearzero = np.any(closetozero)
    phi_indices = np.nonzero(closetozero)[0].tolist()
    
    return any_nearzero, phi_indices, phi_arr_np
      
def filter_phiq(impact_indices, phi_indices, phi_arr_np):
    '''Simultaneous impact must be considered for this project, as the 
    user interaction means initial conditions cannot be pre-set such that
    no simultaneous impacts occur. 
    
    In the case of simultaneous impact of 2 cubes of the same size, there are
    potential phi(q) for indices impacting walls that approach zero even when 
    no impact is occuring at those vertices. Ex: for an exact head-on collision
    of two blocks [ ][ ], the top left vertex of box 1 is "at" the vertical boundary
    of the second box, even though no impact is occurring.
    
    This function filters the indices of phi(q) that are near zero and returns only
    the indices of phi(q) near zero that correspond to impact conditions (evaluated in
    impact_indices) that have been satisfied.
    
    Args:
    - phi_indices    (NP array): passed from phi_nearzero()
    - impact_indices (NP array): passed from impact_condition()
    
    Returns:
    - valid_phiq     (NP array): a subset of phi_indices that corresponds to an 
                                    element of impact_indices
    '''
    phi_indices = np.array(phi_indices)
    impact_indices = np.array(impact_indices)
    inds = np.in1d(phi_indices//4, impact_indices) #evaluates whether elements of phi_indices
                                                    #are related to an impact condition, T/F
    c = np.array(np.nonzero(inds)[0]) #turns locations of these True values to indices
    valid_phiq_indices = phi_indices[c]  
    
    #find location of min valid phi(q)
    valid_phiq = phi_arr_np[valid_phiq_indices]
    argmin = valid_phiq_indices[np.argmin(abs(valid_phiq))]

    #print("\nFilter_Phiq debug:")
    print(f"Impact_indices: {impact_indices}")
    print(f"Phi_indices: {phi_indices}")
    print(f"\nphi_arr_np: \n{phi_arr_np}")
    print(f"Valid_phiq_indices: {valid_phiq_indices}")
    print(f"Argmin: {argmin}")

    #returns the phi(q) equations that both evaluate to ~0 and
    #are related to an impact condition that has been met.
    #returns location of the min phi(q) as well, for knowing which one to apply
    return valid_phiq_indices, argmin          
                                                   

def gen_sym_subs(q, qd_q):
    '''
    Makes three sets of symbolic variables for use in the impact equations.
    Inputs:
    - q: our state vector. ex: [theta1 theta2 theta3]
    - qd_q: our state vector, plus velocities. must have velocities first.
        ex: [theta1d theta2d theta3d theta1 theta2 theta3]
    
    Returns:
    - q_subs: a dictionary of state variables and their "q_1" and "qd_1"
        representations for use in calculation of the impact symbolic equations
    - q_taup_subs: a dictionary that can replace "q_1" and "qd_1" with 
        "q_1^{tau+}" and "qd_1^{tau+}" for solving for the impact update
    - q_taum_subs: ^same as above, but for tau-minus   
    '''

    #enforce that qd_q, which might get confused for q_ext, has derivatives first
    #and other state variables second
    qd_q = sym.Matrix(qd_q).reshape(1, len(qd_q)).tolist()[0]
    for i in range(len(qd_q)-1):
        curr = qd_q[i]
        next = qd_q[i+1]
        if not curr.is_Derivative and next.is_Derivative:
            raise Exception("Gen_sym_subs(): qd_q must have derivatives first")

    #create symbolic substitutions for each element in state array
    sym_q_only = [sym.symbols(f"q_{i+1}") for i in range(len(q))]
    sym_qd = [sym.symbols(f"qd_{i+1}") for i in range(len(q))]
    sym_q = sym_qd + sym_q_only

    # - Define substitution dicts for q at tau+ and q at tau-. We may need
    #the list form as well for later substitutions, so return that as well.
    q_taum_list   = [sym.symbols(f"q_{i+1}")    for i in range(len(q))]
    qd_taum_list  = [sym.symbols(f"qd_{i+1}^-") for i in range(len(q))]
    qd_taup_list  = [sym.symbols(f"qd_{i+1}^+") for i in range(len(q))]

    q_state_dict  = {qd_q[i]: sym_q[i]          for i in range(len(qd_q))}
    qd_taum_dict  = {sym_q[i] : qd_taum_list[i] for i in range(len(q))}
    qd_taup_dict  = {sym_q[i] : qd_taup_list[i] for i in range(len(q))}

    return q_state_dict, qd_taum_dict, qd_taup_dict, \
            q_taum_list, qd_taum_list, qd_taup_list

def impact_symbolic_eqs(phi, lagrangian, q, q_subs):
    '''Takes the impact condition phi, Lagrangian L, and state vector
    q, and returns the expressions we use to evaluate for impact.

    Returns, in order: dL_dqdot, dphi_dq, (dL_dqdot * qdot) - L(q,qdot), 
    '''
    t = sym.symbols(r't')
    qd = q.diff(t)
    
    #define dL_dqdot before substitution
    L_mat = sym.Matrix([lagrangian])
    dL_dqd = L_mat.jacobian(qd)
    
    #define dPhi/dq before substitution
    phi_mat = sym.Matrix([phi])
    dphi_dq = phi_mat.jacobian(q)
    
    #define third expression
    dL_dqd_dot_qd = dL_dqd.dot(qd)
    expr3 = dL_dqd_dot_qd - lagrangian
    
    '''
    at this point the equations are in terms of the
    state variables, x,y, theta1, ...

    convert them into simplified versions "q1, q2, q3, ..."
    for ease of computing the difference between q_tau+ and q_tau- 
    '''
    expr_a = dL_dqd.subs(q_subs)
    expr_b = dphi_dq.subs(q_subs)
    expr_c = expr3.subs(q_subs)
    
    return [expr_a, expr_b, expr_c]

def gen_impact_eqns(phiq_list_sym, lagrangian, q, const_subs):
    '''Methodically calculate all the possible impact updates 
    for the two boxes, using the impact equations derived in class.

    Arguments:
    - phiq_list: 32x0 list of symbolic equations for possible impacts
    - lagrangian: symbolic Lagrangian
    - q: state vector, 6x1 Sympy Matrix
    - const_subs: dictionary of substitutions for m, g, L, w
    '''
    lamb = sym.symbols(r'\lambda')
    t = sym.symbols(r't')

    #qd_q is similar to q_ext, only derivatives come first so that 
    #substitution works properly
    qd_q = sym.Matrix([sym.Matrix(q.diff(t)), q])

    #substitution dictionaries and lists for use in calculating impact
    #update equations
    q_state_dict, qd_taum_dict, qd_taup_dict, \
        q_taum_list, qd_taum_list, qd_taup_list = gen_sym_subs(q, qd_q)

    impacts_eqns_list = []
    for phi in tqdm(phiq_list_sym):
        dL_dqd, dphi_dq, hamiltonian_term = \
            impact_symbolic_eqs(phi,lagrangian, q, q_state_dict)

        lamb_dphi_dq = lamb * dphi_dq

        dL_dqdot_eqn = \
            dL_dqd.subs(qd_taup_dict) \
            - dL_dqd.subs(qd_taum_dict) \
            - lamb_dphi_dq

        hamiltonian_eqn = \
            hamiltonian_term.subs(qd_taup_dict) \
            - hamiltonian_term.subs(qd_taum_dict) \
    
        #sub in m, g, L, w
        dL_dqdot_eqn    = dL_dqdot_eqn.subs(   const_subs)
        hamiltonian_eqn = hamiltonian_eqn.subs(const_subs)

        #these need to be simplified or else they're uninterpretable
        #print(f"Simplifying impact equations. Started at: {datetime.now()}")
        #t0 = time.time()
        dL_dqdot_eqn = sym.simplify(dL_dqdot_eqn)
        hamiltonian_eqn = sym.simplify(hamiltonian_eqn)
        dL_dqdot_eqn = dL_dqdot_eqn.T
        #tf = time.time()

        #print(f"\nImpacts simplify: \nElapsed: {round(tf - t0, 2)} seconds")
        eqns_matrix = dL_dqdot_eqn.row_insert( len(q), sym.Matrix([hamiltonian_eqn]))
        impacts_eqns_list.append(eqns_matrix)
        
    #save outcome to a file so we can load it during simulation
    #dill_dump('../dill/impacts_eqns_32x.dill',impacts_eqns_list)
    return impacts_eqns_list

def impact_update(s, impact_eqs, sol_vars):
    '''Once an impact has been detected, apply the necessary
    impact update based on which equation has just occurred.

    Args:
    - s: full state of system. Contains x, y, theta1, theta2,
        phi1, phi2, and their derivatives.
    - impact_eqs: the symbolic impact equations that need to be solved.
    - sol_vars: list of variables we're solving for. qd1_tau+, qd2_tau+, ..., lambda.
    '''
    curr_state_subs = {**{sym.symbols(f"q_{i+1}") : s[i] for i in range(6)},
                         **{sym.symbols(f"qd_{i+1}^-") : s[i+6] for i in range(6)}}
    impact_eqs_curr = impact_eqs.subs(curr_state_subs)


    #print("\nImpact update debug:")
    #print(f"State s, 6-12: {s[6:12]}")
    #print(f"Initial guess: {init_guess}")

    #code is based on Jake's code from discussion page
    #would be great if I could log the impacts separately and maintain tqdm running
    attempts = 10
    init_guess = -1 * np.append(s[6:12],[0])
    solns_list = [0]*attempts
    lamb_val_arr = np.zeros(10)
    soln = None

    for i in range(attempts):
        #try:
        curr_soln = sym.nsolve(impact_eqs_curr, sol_vars, init_guess, dict = True, verify = False)[0]
        solns_list[i] = curr_soln
        lamb_val_arr[i] = curr_soln[lamb]

        #print(f"Iteration {i}: lambda {curr_soln[lamb]}")
        #print(curr_soln)
        #if abs(curr_soln[lamb]) > 1e-9:
        #    #print(curr_soln)
        #    soln = curr_soln
        #    print(f"Solution selected: lambda = {soln[lamb]}")

        #    break       
        if i >= 5:
            ind = np.argmax(abs(lamb_val_arr))
            soln = solns_list[ind]
            lamb_val = lamb_val_arr[ind]
            #print(f"\nSolution selected: lambda = {soln[lamb]}")

    
        #except Exception as e:
        #    print(f"Nsolve threw an error: {e}")
        #init_guess = -0.5*init_guess
        init_guess = -1.5*init_guess

    print("\nArray of solns:")
    print(solns_list)
    #I'm gonna assume order of dict values stays as qd_tau1+, qd_tau2+, ... qd_tau6+ but
    #this may be something to debug if my impacts look weird
    if soln:
        del soln[lamb]
        qd_tauplus = np.array(list(soln.values())).astype('float')
        new_state = np.append(s[0:6], qd_tauplus )
        return new_state
    else:
        print("No solution found by nsolve")
        return s


def simulate_impact(t_span, dt, ICs, integrate, dxdt, impact_condition, impact_update):
    '''
    simulate(), but with an extra framework for detecting impact
    
    Inputs:
    - t_span: 2-elem array [to, tf]
    - dt: timestep, float
    - ICs: n-dim array with the initial state of system
    - integrate: type "function". for our integration scheme (usually RK4 or Euler)
    - dxdt: type "function". our derivative function, used to calculate next statew
    - impact_condition: type "function". takes in state s, returns True if particle passes through a boundary
    - impact_update: type "function". takes in s at tau-, returns the state of the system at tau+
    
    Returns:
    - traj_array: an nxm array, where m = length of the time vector and n = # of variables in state s
    '''
    
    #load in symbolic equations from previous solves
    impact_eqns_0_32 = dill_load('../dill/impact_eqns_0_32.dill')
    #atol = 1E-4 # for phiq_near_zero
    #atol = 3E-3 # for phiq_near_zero
    #atol = 1E-2 # for phiq_near_zero
    #atol = 5E-2 # for phiq_near_zero

    atol = 5E-1 # for phiq_near_zero


    #qd_q is similar to q_ext, only derivatives come first so that 
    #substitution works properly
    t = sym.symbols(r't')
    qd_q = sym.Matrix([sym.Matrix(q.diff(t)), q])
    
    #describe variables we're solving for - qd1_tau+, qd2_tau+, ... lambda 
    lamb = sym.symbols(r'\lambda')
    _, _, _, _, _, qd_taup_list = gen_sym_subs(q, qd_q)
    sol_vars = qd_taup_list
    sol_vars.append(lamb)

    #t_array = np.arange(t_span[0], t_span[1], dt)
    #traj_array = np.zeros([len(ICs), len(t_array)])
    #traj_array[:,0] = ICs

    #print("Simulating system with impacts...")
    #for i in tqdm(range(len(t_array) - 1)):
    
    #    #get current value of s
    #    t = t_array[i]
    #    s = traj_array[:,i]

    #    #calculate s for next timestep
    #    s_next = integrate(dxdt, s, t, dt)
    #    s_twodt = integrate(dxdt, s_next, t+dt, dt)
    #    s_check = s_next[:]

    #    #check if impact has occurred, either at curr. timestep or 2 timesteps from now
    #    impact_dt,    impact_indices       = impact_condition(s_next[0:6])
    #    impact_twodt, impact_indices_twodt = impact_condition(s_twodt[0:6])

    #    ##if system impacts two timesteps from now, apply the impact update using
    #    ##that data.
    #    if impact_twodt and not impact_dt:
    #        impact_dt = impact_twodt
    #        impact_indices = impact_indices_twodt[:]
    #        s_check = s_twodt[:]


    #    if (impact_dt):
    #        '''This is designed to alter the velocity of the particle
    #        just before impact. If we applied the impact update after impact
    #        (same position, changed velocity), there's a chance the objects 
    #        would stay stuck inside each other.
    #        ''' 
            
    #        #find phi(q) we can apply to the system. choose one to apply
    #        #any_nearzero, phi_indices, phi_arr_np = phi_nearzero(s_next[0:6], atol)
    #        any_nearzero, phi_indices, phi_arr_np = phi_nearzero(s_check[0:6], atol)

    #        valid_phiq_indices, argmin = filter_phiq(impact_indices, phi_indices, phi_arr_np)
            
    #        #this is a case I eventually want to figure out
    #        if len(valid_phiq_indices) == 0:
    #            print("Invalid phi(q)/impact condition combination") #throw an error in the future
    #        else:
    #            #index = valid_phiq_indices[0]
    #            #impact_eqs = impact_eqns_0_32[index]
    #            impact_eqs = impact_eqns_0_32[argmin]

    #            #solve for next state, using numerical nsolve() on symbolic expressions
    #            #s_alt = impact_update(s_next, impact_eqs, sol_vars)
    #            s_alt = impact_update(s, impact_eqs, sol_vars)


    #            #if particle is still inside block at next timestep, integrate for 2dt
    #            #impact_next, _ = impact_condition(s_alt[0:6])
    #            #if impact_next:    
    #            #    s_next = integrate(dxdt, s_alt, t, 2*dt)
    #            #else:
    #            #    s_next = integrate(dxdt, s_alt, t, dt)
    #            s_next = integrate(dxdt, s_alt, t, dt)


    #            #impose constraints on velocity to prevent particle from diverging to infinity
    #            #s_next = np.append(np.clip(s_next[0:6], -100, 100), np.clip(s_next[6:12], -10, 10))
    #            s_next = np.append(  
    #                    np.clip(s_next[0:6], -abs(s[0:6]) - 0.2, abs(s[0:6]) + 0.2), \
    #                    np.clip(s_next[6:12], -10, 10)
    #                )


    #    #apply update to trajectory vector            
    #    traj_array[:, i+1] = s_next

    #save results to a CSV file for use in animation. This needs to be removed before submission.
    pd.DataFrame(traj_array.T).to_csv('../csv/q_array_impacts.csv', header=None, index=None)
    return traj_array


##SOME GLOBAL VARIABLES
xy_coords_list = convert_coords_to_xy()
vertices_list_np = [sym.lambdify(q, expr.subs(subs_dict)) for expr in xy_coords_list] #subs_dict =  constants
impact_eqns_0_32 = dill_load('../dill/impact_eqns_0_32.dill')
lamb = sym.symbols(r'\lambda')


#things to only be calculated here
if __name__ == '__main__':

    #---------global variables for use in other files-------------#

    #calculate_sym_vertices()
   
    print("Setting up Euler-Lagrange Eqns and dxdt()...")

    #F_eqs_array = np.array([
    #    #lambda s,t: 0, #F_x
    #    lambda s,t: -5*s[6], #F_x - damping term applied to vel.
    #    lambda s,t: 19.62 -10*s[7], #F_y
    #    lambda s,t: 2*s[8], #F_theta1
    #    lambda s,t: 2*s[8], #F_theta2
    #    lambda s,t: 0, #F_phi1
    #    lambda s,t: 0, #F_phi2
    #])

    F_eqs_array = np.array([
        #lambda s,t: 0, #F_x
        lambda s,t: -5*s[6], #F_x - damping term applied to vel.
        lambda s,t: 19.62 - 10*s[7], #F_y
        lambda s,t: 0, #F_theta1
        lambda s,t: 0, #F_theta2
        lambda s,t: 0, #F_phi1
        lambda s,t: 0, #F_phi2
    ])

    #exciting theta1 and theta1 at const 2 or 10 doesn't work

    dxdt = construct_dxdt(F_eqs_array)

    dt = 0.005
    t_span = [0, 5]

    theta0 = np.pi/4
    init_posns = [1, 1, theta0, -theta0, 2*np.pi/3, np.pi/4]
    init_velocities = [0, 0, -1, 1, 6, -4]
    ICs = init_posns + init_velocities
    #th0 = 0.2315 #initial theta such that only one vertex is impacting
    #init_posns = [0,1, th0, -th0, 0, 0]
    #init_velocities = [0, 0, 0, 0, 0, 0]
    #ICs = init_posns + init_velocities


    print("Preparing numerical impact conditions...")
    lamb = sym.symbols(r'\lambda')
    xy_coords_list = convert_coords_to_xy()
    vertices_list_np = [sym.lambdify(q, expr.subs(subs_dict)) for expr in xy_coords_list] #subs_dict =  constants

    simulate_impact(t_span, dt, ICs, rk4, dxdt, impact_condition, impact_update)
