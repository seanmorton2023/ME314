import numpy as np
import sympy as sym
import dill
import time
from datetime import datetime
from tqdm import tqdm

from geometry import *
from helpers import *
from el_equations import compute_lagrangian

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
    v1bar = w*sym.Matrix([ 1/sym.sqrt(2),  1/sym.sqrt(2), 0, 1])
    v2bar = w*sym.Matrix([-1/sym.sqrt(2),  1/sym.sqrt(2), 0, 1])
    v3bar = w*sym.Matrix([-1/sym.sqrt(2), -1/sym.sqrt(2), 0, 1])
    v4bar = w*sym.Matrix([ 1/sym.sqrt(2), -1/sym.sqrt(2), 0, 1])

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
    
    impact_conds = np.array([
        -wval/2.0 < v11x_B2_np(s) < wval/2.0   and   -wval/2.0 < v11y_B2_np(s) < wval/2.0,
        -wval/2.0 < v12x_B2_np(s) < wval/2.0   and   -wval/2.0 < v12y_B2_np(s) < wval/2.0,
        -wval/2.0 < v13x_B2_np(s) < wval/2.0   and   -wval/2.0 < v13y_B2_np(s) < wval/2.0,
        -wval/2.0 < v14x_B2_np(s) < wval/2.0   and   -wval/2.0 < v14y_B2_np(s) < wval/2.0,
        
        -wval/2.0 < v21x_B1_np(s) < wval/2.0   and   -wval/2.0 < v21y_B1_np(s) < wval/2.0,
        -wval/2.0 < v22x_B1_np(s) < wval/2.0   and   -wval/2.0 < v22y_B1_np(s) < wval/2.0,
        -wval/2.0 < v23x_B1_np(s) < wval/2.0   and   -wval/2.0 < v23y_B1_np(s) < wval/2.0,
        -wval/2.0 < v24x_B1_np(s) < wval/2.0   and   -wval/2.0 < v24y_B1_np(s) < wval/2.0,     
    ])
    
    #find any impact conditions that have been met
    impact_met = np.any(impact_conds)
    impact_indices = np.nonzero(impact_conds)[0].tolist() #indices where true
    
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
        phi_arr_np = np.append(phi_arr_np, vertices_list_np[i](s) + w_num/2.0)
        phi_arr_np = np.append(phi_arr_np, vertices_list_np[i](s) - w_num/2.0)

    #we're interested in which of the phi conditions were evaluated at close to 0, so return the
    #list of indices close to 0
    closetozero = np.isclose( phi_arr_np, np.zeros(phi_arr_np.shape), atol=atol )
    
    #this gives the indices of phi close to 0
    any_nearzero = np.any(closetozero)
    phi_indices = np.nonzero(closetozero)[0].tolist()
    
    return any_nearzero, phi_indices
      
def filter_phiq(impact_indices, phi_indices):
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
    inds = np.in1d(phi_indices//4, impact_indices) #evaluates whether elements of phi_indices
                                                    #are related to an impact condition, T/F
    c = np.array(np.nonzero(inds)[0]) #turns locations of these T values to indices
    valid_phiq = phi_indices[c]  
    return valid_phiq                 #returns the phi(q) equations that both evaluate to ~0 and
                                       #are related to an impact condition that has been met

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

    # - Define substitution dicts for q at tau+ and q at tau-,
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

def gen_impact_updates(phiq_list_sym, lagrangian, q, const_subs):
    '''Methodically calculate all the possible impact updates 
    for the two boxes, using the impact equations derived in class.

    Arguments:
    - phiq_list: 32x0 list of symbolic equations for possible impacts
    - lagrangian: symbolic Lagrangian
    - q: state vector, 6x1 Sympy Matrix
    - const_subs: dictionary of substitutions for m, g, L, w
    '''
    lamb = sym.symbols(r'\lambda')

    #qd_q is similar to q_ext, only derivatives come first so that 
    #substitution works properly
    qd_q = sym.Matrix([sym.Matrix(q.diff(t)), q])

    #substitution dictionaries and lists for use in calculating impact
    #update equations
    q_state_dict, q_taum_dict, q_taup_dict, \
        q_taum_list, qd_taum_list, qd_taup_list = gen_sym_subs(q, qd_q)

    #for phi in phiq_list_sym:
    sample_phi = phiq_list_sym[0]
    dL_dqd, dphi_dq, hamiltonian_term = \
            impact_symbolic_eqs(sample_phi,lagrangian, q, q_state_dict)

    '''
    lamb_dphi_dq = lamb * dphi_dq

    dL_dqdot_eqn = \
        dL_dqd.subs(q_taup_dict) \
        - dL_dqd.subs(q_taum_dict) \
        - lamb_dphi_dq

    hamiltonian_eqn = \
        hamiltonian_term.subs(q_taup_dict) \
        - hamiltonian_term.subs(q_taum_dict) \
    
    #sub in m, g, L, w
    dL_dqdot_eqn    = dL_dqdot_eqn.subs(   const_subs)
    hamiltonian_eqn = hamiltonian_eqn.subs(const_subs)

    #hamiltonian_eqn = hamiltonian_eqn.simplify()
    '''

    '''
    #these need to be simplified or else they're uninterpretable
    print(f"Simplifying impact equations. Started at: {datetime.now()}")
    t0 = time.time()
    dL_dqdot_eqn = sym.simplify(dL_dqdot_eqn)
    hamiltonian_eqn = sym.simplify(hamiltonian_eqn)
    dL_dqdot_eqn = dL_dqdot_eqn.T
    tf = time.time()

    print(f"\nImpacts simplify: \nElapsed: {round(tf - t0, 2)} seconds")

    #save results for the first round of this to a file so 
    #we don't have to worry about saving + reloading
    dill_dump('../dill/impacts_hamiltonian_v1.dill', hamiltonian_eqn)
    dill_dump('../dill/impacts_dL_dqdot_eq_v1.dill', dL_dqdot_eqn)
    '''

    hamiltonian_eqn = dill_load('../dill/impacts_hamiltonian_v1.dill')
    dL_dqdot_eqn =    dill_load('../dill/impacts_dL_dqdot_eq_v1.dill')

    #set up equations to be solved
    #insert the hamiltonian at the (n)th row in 0-based indexing, i.e. add onto end of matrix
    eqns_matrix = dL_dqdot_eqn.row_insert( len(q), sym.Matrix([hamiltonian_eqn]))
    
    #solve for the values of qdot and lambda
    sol_vars = qd_taup_list
    sol_vars.append(lamb)

    print(f"Solving impact equations. Started at: {datetime.now()}")
    t0 = time.time()
    solns = sym.solve(eqns_matrix, sol_vars, dict = True, simplify = False, manual=True)
    tf = time.time()

    print(f"\nImpacts solve: \nElapsed: {round(tf - t0, 2)} seconds")
    print(solns)
    dill_dump('../dill/impacts_solns_v1.dill', solns)


    pass

#things to only be calculated here
if __name__ == '__main__':

    #---------global variables for use in other files-------------#

    #calculate_sym_vertices()
    print("Preparing Lagrangian and impact constraints...")
    xy_coords_list = convert_coords_to_xy()
    vertices_list_np = [sym.lambdify(q, expr) for expr in xy_coords_list]
    phiq_list_sym = calculate_sym_phiq(xy_coords_list)
    lagrangian = compute_lagrangian()

    gen_impact_updates(phiq_list_sym, lagrangian, q, subs_dict) #subs_dict = constants

    #display these on Jupyter notebook, so we need to save them to file
    #dill_dump('../dill/phiq_list.dill', phiq_list_sym)

    #print("Sample expression of vertex x coordinate:")           
    #display(v21x_B1)

