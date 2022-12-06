def animate_traj_array(q_array, win_dims:tuple, world_dims:tuple,
                       L=1, w=1/3.0, t=10):
    '''This function tales an existing array of data and plots it, frame by
    frame, using the Tkinter canvas. The purpose of this function is to debug
    the animation and to visualize the framerate of updates on the Tkinter
    canvas.
    
    Arguments:
    - q_array: array of state values
    - L, w: numerical values of length + width of string + box
    - t: length of simulation time.
    - win_dims (win_width, win_height): dimensions in pixels of Tk Canvas
    - world_dims (width, height): dimensions in world units of visible Canvas
    '''  
    win_width, win_height = win_dims
    width, height = world_dims

    #define constant matrices
    
    vertices_mat = w * np.array([
        [ np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1],
        [-np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1],
        [-np.sqrt(2)/2, -np.sqrt(2)/2, 0, 1],
        [ np.sqrt(2)/2, -np.sqrt(2)/2, 0, 1],
        [ np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1], #add first vertex onto end of matrix again so line wraps around
    ]).T #in "bar" form  so they can be multiplied by trans. matrix
    
    line_coords_mat = np.array([
        [0,  0, 0, 1],
        [0, -L, 0, 1],
    ]).T
    
    #let frame GUI be the coordinates as seen on the GUI,
    #frame r be the frame at GUI coords (0,0) with axes in same direction
    #as frame s
    GrGUI = np.array([
        [width/win_width,    0, 0, 0],
        [0, -height/win_height, 0, 0],
        [0,                      0, 1, 0],
        [0,                      0, 0, 1]
    ]) #note: not a rotation; instead is a scaling/flipping operation
    
    Grs = SOnAndRnToSEn(np.identity(3), [-width/2, height/2, 0])
    GsGUI = np.dot(InvSEn(Grs), GrGUI)
    
    ###
    
    for i, q in enumerate(q_array):
        box1_vert_gui, box2_vert_gui, line1_coords_gui, line2_coords_gui = \
            get_GUI_coords(q, line_coords_mat, vertices_mat, GsGUI, L, w)
        
        #note: add first vertices of box coords onto array as well
        
        if i == 0:
            #create objects on the canvas
            canvas.create_line(*box1_vert_gui, tag='box1', fill='gray')
            canvas.create_line(*box2_vert_gui, tag='box2', fill='gray')
            canvas.create_line(*line1_coords_gui, tag='line1', fill='gray')
            canvas.create_line(*line2_coords_gui, tag='line2', fill='gray')
            
        else:
            #update positions of the objects by tags
            canvas.coords('box1', *box1_vert_gui)
            canvas.coords('box2', *box2_vert_gui)
            canvas.coords('line1', *line1_coords_gui)
            canvas.coords('line2', *line2_coords_gui)


def test_gen_impact_updates():
    lamb = sym.symbols(r'\lambda')
    t = sym.symbols(r't')

    #qd_q is similar to q_ext, only derivatives come first so that 
    #substitution works properly
    qd_q = sym.Matrix([sym.Matrix(q.diff(t)), q])

    #substitution dictionaries and lists for use in calculating impact
    #update equations
    q_state_dict, qd_taum_dict, qd_taup_dict, \
        q_taum_list, qd_taum_list, qd_taup_list = gen_sym_subs(q, qd_q)

    #for phi in phiq_list_sym:
    sample_phi = phiq_list_sym[0]
    dL_dqd, dphi_dq, hamiltonian_term = \
            impact_symbolic_eqs(sample_phi,lagrangian, q, q_state_dict)

    hamiltonian_eqn = dill_load('../dill/impacts_hamiltonian_v1.dill')
    dL_dqdot_eqn =    dill_load('../dill/impacts_dL_dqdot_eq_v1.dill')

    #set up equations to be solved
    #insert the hamiltonian at the (n)th row in 0-based indexing, i.e. add onto end of matrix
    #replace sinusoid terms with dummy variables
    eqns_matrix = dL_dqdot_eqn.row_insert( len(q), sym.Matrix([hamiltonian_eqn]))
    sinusoidal_subs, sinusoidal_subs_inv = gen_sinusoid_subs(eqns_matrix)
    eqns_matrix_poly = eqns_matrix.subs(sinusoidal_subs)

    #solve for the values of qdot and lambda
    sol_vars = qd_taup_list
    sol_vars.append(lamb)

    #solve impacts equations as a set of polynomials
    #eqns_matrix_poly = eqns_matrix_poly.evalf()

    print(f"Solving impact equations. Started at: {datetime.now()}")
    t0 = time.time()
    solns = sym.solve(eqns_matrix_poly, sol_vars, dict = True, simplify = False, minimal = True)
    tf = time.time()

    print(f"\nImpacts solve: \nElapsed: {round(tf - t0, 2)} seconds")
    print(solns)
    dill_dump('../dill/impacts_solns_v1.dill', solns)
    #dill_dump('../dill/impacts_solns_nonlinsolve.dill', out_set)



def gen_sinusoid_subs(eqns_matrix):
    '''Takes a matrix of expressions, and return a set of all sinusoidal terms
    contained within that matrix.

    Motivation: one good candidate for solving the impact equations is sympy.nonlinsolve(),
        which cannot take in sinusoidal terms - it can only 

    Returns: sinusoidal terms substitution dict (with placeholders "c1, c2, ..." as sub)
        and its inverse (mapping from dummy variables c1, c2, ... back to sin(...) ).
    '''
    factors_list = np.array([])
    #generally I should avoid nested for loops but this is a small # of computations
    for row in eqns_matrix:
        for factor in row.as_ordered_terms():
            factors_list = np.append(factors_list, list(factor.as_coeff_mul()[-1]) )

    factors_dict = {}
    for factor in factors_list:
        if factor in factors_dict.keys():
            factors_dict[factor] += 1
        else:
            factors_dict[factor] = 1

    #make a loop structure for creating the simplest set of factors
    factors_list = np.array(list(factors_dict.keys()))
    condition = np.any([x.is_Mul or x.is_Add or x.is_Pow for x in factors_list])

    while condition:
        factors_dict = decompose_factors_dict(factors_dict) #from helpers.py
        factors_list = np.array(list(factors_dict.keys()))
        condition = np.any([x.is_Mul or x.is_Add or x.is_Pow for x in factors_list])
    
    #get list of sin() and cos() terms
    sinusoids = [f for f in factors_list if (type(f) == sym.cos or type(f) == sym.sin)]

    #make symbolic substitutions for sinusoids in the impact equations - this 
    #will make the impact equations into polynomials, solvable by nonlinsolve()
    sinusoid_subs = {sinusoids[i] : sym.symbols(f"c_{i+1}") for i in range(len(sinusoids))}
    sinusoid_subs_inv = {val : key for key, val in sinusoid_subs.items()}

    return sinusoid_subs, sinusoid_subs_inv

