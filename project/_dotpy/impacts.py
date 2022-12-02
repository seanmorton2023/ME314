import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

def impact_condition(s):
    '''Contains and evaluates an array of impact conditions for the current
    system, at the current state s. 
    
    Returns: a logical true/false as to whether any impact condition was met;
        list of indices of impact conditions that were met
    '''
    
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
    
#impact conditions

#define positions of vertices
v1bar = w*sym.Matrix([ 1/sym.sqrt(2),  1/sym.sqrt(2), 0, 1])
v2bar = w*sym.Matrix([-1/sym.sqrt(2),  1/sym.sqrt(2), 0, 1])
v3bar = w*sym.Matrix([-1/sym.sqrt(2), -1/sym.sqrt(2), 0, 1])
v4bar = w*sym.Matrix([ 1/sym.sqrt(2), -1/sym.sqrt(2), 0, 1])

GB1B2 = InvSEn(GsB1) @ GsB2
GB2B1 = InvSEn(GB1B2)

#subscripts: let v12 mean "the 2nd vertex of the 1st body"\
#the home frame of v1n is in body 1; impact cond. requires posn in 2nd body frame
v11_B2 = sym.simplify(GB2B1 @ v1bar)
v12_B2 = sym.simplify(GB2B1 @ v2bar)
v13_B2 = sym.simplify(GB2B1 @ v3bar)
v14_B2 = sym.simplify(GB2B1 @ v4bar)

#size of blocks is the same, so posn of vertices v2n in B2 frame is same
#as posn of v1n in B1 frame
v21_B1 = sym.simplify(GB1B2 @ v1bar)
v22_B1 = sym.simplify(GB1B2 @ v2bar)
v23_B1 = sym.simplify(GB1B2 @ v3bar)
v24_B1 = sym.simplify(GB1B2 @ v4bar)

#find x and y components of posn
v11x_B2, v11y_B2 = v11_B2
v12x_B2, v12y_B2 = v12_B2
v13x_B2, v13y_B2 = v13_B2
v14x_B2, v14y_B2 = v14_B2

v21x_B1, v21y_B1 = v21_B1
v22x_B1, v22y_B1 = v22_B1
v23x_B1, v23y_B1 = v23_B1
v24x_B1, v24y_B1 = v24_B1
#---#

print("Sample expression of vertex x coordinate:")           
display(v21x_B1)

#substitute in values 
v11x_B2_np = sym.lambdify(q, v11x_B2)
v11y_B2_np = sym.lambdify(q, v11y_B2)

v12x_B2_np = sym.lambdify(q, v12x_B2)
v12y_B2_np = sym.lambdify(q, v12y_B2)

v13x_B2_np = sym.lambdify(q, v13x_B2)
v13y_B2_np = sym.lambdify(q, v13y_B2)

v14x_B2_np = sym.lambdify(q, v14x_B2)
v14y_B2_np = sym.lambdify(q, v14y_B2)

###

v21x_B1_np = sym.lambdify(q, v21x_B1)
v21y_B1_np = sym.lambdify(q, v21y_B1)

v22x_B1_np = sym.lambdify(q, v22x_B1)
v22y_B1_np = sym.lambdify(q, v22y_B1)

v23x_B1_np = sym.lambdify(q, v23x_B1)
v23y_B1_np = sym.lambdify(q, v23y_B1)

v24x_B1_np = sym.lambdify(q, v24x_B1)
v24y_B1_np = sym.lambdify(q, v24y_B1)


# In[17]:


#define full set of impact conditions
#wval = subs_dict[w]