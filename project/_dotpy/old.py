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