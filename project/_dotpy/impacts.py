import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

from geometry import *
from helpers import *

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
    f_list = np.array([
        v11x_B2_np, v11y_B2_np,
        v12x_B2_np, v12y_B2_np,
        v13x_B2_np, v13y_B2_np,
        v14x_B2_np, v14y_B2_np,

        v21x_B1_np, v21y_B1_np,
        v22x_B1_np, v22y_B1_np,
        v23x_B1_np, v23y_B1_np,
        v24x_B1_np, v24y_B1_np,
    ])
    
    #apply upper and lower bound condition to all vertices, in x and y directions
    phi_arr_np = np.array([])
    for i in np.arange(0, len(f_list)):
        phi_arr_np = np.append(phi_arr_np, f_list[i](s) + wval/2.0)
        phi_arr_np = np.append(phi_arr_np, f_list[i](s) - wval/2.0)

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
    filepath = '../data/vertices_coords_list.dill'
    dill_dump(filepath, vertices_coords_list)
    print(f"Vertices coords saved to {filepath}.")


def convert_coords_to_xy():
    '''Take the symbolic coordinates we found for the two boxes and 
    split them into x and y components.
    '''
    vertices_coords_list = dill_load('../data/vertices_coords_list.dill')
    vertices_xy_list = []
    for coord in vertices_coords_list:
        coordx, coordy, _, _ = coord
        vertices_xy_list.append([coordx, coordy])

    #flatten
    vertices_list_sym = np.array(vertices_xy_list).flatten().tolist()
    return vertices_list_sym

    ##subscripts: let v12 mean "the 2nd vertex of the 1st body"\
    ##the home frame of v1n is in body 1; impact cond. requires posn in 2nd body frame
    #v11_B2 = sym.simplify(GB2B1 @ v1bar)
    #v12_B2 = sym.simplify(GB2B1 @ v2bar)
    #v13_B2 = sym.simplify(GB2B1 @ v3bar)
    #v14_B2 = sym.simplify(GB2B1 @ v4bar)

    ##size of blocks is the same, so posn of vertices v2n in B2 frame is same
    ##as posn of v1n in B1 frame
    #v21_B1 = sym.simplify(GB1B2 @ v1bar)
    #v22_B1 = sym.simplify(GB1B2 @ v2bar)
    #v23_B1 = sym.simplify(GB1B2 @ v3bar)
    #v24_B1 = sym.simplify(GB1B2 @ v4bar)

    #find x and y components of posn
    #v11x_B2, v11y_B2 = v11_B2
    #v12x_B2, v12y_B2 = v12_B2
    #v13x_B2, v13y_B2 = v13_B2
    #v14x_B2, v14y_B2 = v14_B2

    #v21x_B1, v21y_B1 = v21_B1
    #v22x_B1, v22y_B1 = v22_B1
    #v23x_B1, v23y_B1 = v23_B1
    #v24x_B1, v24y_B1 = v24_B1
    
    ##------------------------#
    #vertices_list_sym = [
    #    v11x_B2, v11y_B2,
    #    v12x_B2, v12y_B2,
    #    v13x_B2, v13y_B2,
    #    v14x_B2, v14y_B2,

    #    v21x_B1, v21y_B1,
    #    v22x_B1, v22y_B1,
    #    v23x_B1, v23y_B1,
    #    v24x_B1, v24y_B1        
    #]

    #return vertices_list_sym

###

def calculate_sym_phiq():
    '''
    Calculate the symbolic impact equations Phi(q) for use in 
    applying impact updates to the system. There will be 32 
    phi(q) equations - 2 per possible vertex + side of impact
    combination.

    Saves symbolic phi(q) to a pickled file for loading
    and use in other files, plus saving variables between sessions
    of Python/Jupyter notebook.

    Returns: None; load symbolic phi(q) from file for simplicity
    '''

    phiq_list = []
    for vertex in vertices_list:
        phiq_list.append() #see if we need to make an equality datatype
                            #or if we can just do a-b

    pass



#things to only be calculated here
if __name__ == '__main__':


    print("Sample expression of vertex x coordinate:")           
    display(v21x_B1)

#---------global variables for use in other files-------------#

#vertices_list_np = [sym.lambdify(q, expr) for expr in vertices_list]





#OLD


#v11x_B2_sym, v11y_B2_sym,\
#v12x_B2_sym, v12y_B2_sym,\
#v13x_B2_sym, v13y_B2_sym,\
#v14x_B2_sym, v14y_B2_sym,\

#v21x_B1_sym, v21y_B1_sym,\
#v22x_B1_sym, v22y_B1_sym,\
#v23x_B1_sym, v23y_B1_sym,\
#v24x_B1_sym, v24y_B1_sym  = vertices_list

##substitute in values 
#v11x_B2_np = sym.lambdify(q, v11x_B2_sym)
#v11y_B2_np = sym.lambdify(q, v11y_B2_sym)
##
#v12x_B2_np = sym.lambdify(q, v12x_B2_sym)
#v12y_B2_np = sym.lambdify(q, v12y_B2_sym)
##
#v13x_B2_np = sym.lambdify(q, v13x_B2_sym)
#v13y_B2_np = sym.lambdify(q, v13y_B2_sym)
##
#v14x_B2_np = sym.lambdify(q, v14x_B2_sym)
#v14y_B2_np = sym.lambdify(q, v14y_B2_sym)


#v21x_B1_np = sym.lambdify(q, v21x_B1_sym)
#v21y_B1_np = sym.lambdify(q, v21y_B1_sym)
##
#v22x_B1_np = sym.lambdify(q, v22x_B1_sym)
#v22y_B1_np = sym.lambdify(q, v22y_B1_sym)
##
#v23x_B1_np = sym.lambdify(q, v23x_B1_sym)
#v23y_B1_np = sym.lambdify(q, v23y_B1_sym)
##
#v24x_B1_np = sym.lambdify(q, v24x_B1_sym)
#v24y_B1_np = sym.lambdify(q, v24y_B1_sym)
