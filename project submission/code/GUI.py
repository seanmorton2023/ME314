import tkinter as tk
import numpy as np

from geometry import *
from helpers import *
from impacts import *
from el_equations import *
from plotting_helpers import *

###########

#contains the symbolic impact equations - will be solved using nsolve() during sim
impact_eqns_0_32 = dill_load('../dill/impact_eqns_0_32.dill')

class GUI:

    def __init__(self, win_height, win_width):

        #future improvement: should inherit from the Tk class
        self.root = tk.Tk()
        self.root.title("Clacker Balls Simulation")
        self.canvas = tk.Canvas(self.root, bg="white", height=win_height, width=win_width)
        self.win_height = win_height
        self.win_width = win_width

        #member data we expect to change on each loop
        self.timer_handle = None #set this from the outside once canvas is packed
        self.last_frametime = 0
        self.q_ind = 0
        self.impact_photoID = None
        self.mouse_posn_gui = [win_width//2, win_height//2]
        self.mouse_posn_s = [0,0]

    ###

    #these values are determined externally, but loading them all in __init__
    #would be too long, so do it here

    def load_arrays(self, line_coords_mat, vertices_mat):
        self.line_coords_mat = line_coords_mat
        self.vertices_mat    = vertices_mat

    def load_gui_params(self, L, w, coordsys_len, GsGUI, framerate, photo_filepath):
        self.L = L
        self.w = w
        self.coordsys_len = coordsys_len
        self.GsGUI = GsGUI
        self.framerate_ms = framerate
        self.impact_photoID = draw_image(self.canvas, self.root, \
            (self.win_width//2, self.win_height//2), photo_filepath, size=0,
                  tags='sparks', state='hidden')

    def load_simulation(self, dxdt, t_span, dt, ICs, atol):

        #describe variables we're solving for - qd1_tau+, qd2_tau+, ... lambda 
        sym_t = sym.symbols(r't')
        qd_q = sym.Matrix([sym.Matrix(q.diff(sym_t)), q])
        lamb = sym.symbols(r'\lambda')
        _, _, _, _, _, qd_taup_list = gen_sym_subs(q, qd_q)

        self.sol_vars = qd_taup_list
        self.sol_vars.append(lamb)

        #timescales and tolerance for checking if phi(q) near zero
        self.dt = dt
        self.t_array = np.arange(t_span[0], t_span[1], dt)
        self.traj_array = np.zeros([len(ICs), len(self.t_array)])
        self.traj_array[:,0] = ICs
        self.dxdt = dxdt
        self.atol = atol

    #-----------------------------------#

    def get_GUI_coords(self, q):
        '''
        Takes the present value of the state array and returns the 
        coordinates of the key items on the GUI: the coords of
        the lines for the two strings, and the coords of the boxes
        for the two masses.
    
        Arguments:
        - q: current value of extended state array [q; qdot]
        - line_coords_mat: a 4xn array, n = 2 points per line,
            with the coordinates of lines in their reference frames
        - vertices_mat: a 4x5 array (4 vertices per box, plus the initial
            coordinate repeated) with coordinates of vertices of the boxes
            in their reference frames
        - GsGUI: transformation of points from space frame to GUI frame
            (note: not SE(3) - scaling + mirroring operations)
        - L: length of string
        - w: width of box
    
        Returns:   
        - box1_vert_gui:    cods of object in GUI frame 
        - box2_vert_gui:    coords of object in GUI frame 
        - line1_coords_gui: coords of object in GUI frame 
        - line2_coords_gui: coords of object in GUI frame
        '''
    
        #extract coords
        x, y, theta1, theta2, phi1, phi2 = q[0:6]
        
        #define frames

        #---------------right side---------------#

        Rab = np.array([
            [np.cos(theta2), -np.sin(theta2), 0],
            [np.sin(theta2),  np.cos(theta2), 0],
            [              0,                0, 1],
        ])

        RdB2 = np.array([
            [np.cos(phi2), -np.sin(phi2), 0],
            [np.sin(phi2),  np.cos(phi2), 0],
            [            0,              0, 1],
        ])

        p_bd = np.array([0, -self.L, 0])

        Gab = SOnAndRnToSEn(Rab, [0, 0, 0])
        Gbd = SOnAndRnToSEn(np.eye(3), p_bd)
        GdB2 = SOnAndRnToSEn(RdB2, [0, 0, 0])

        Gsa = SOnAndRnToSEn(np.eye(3), [x, y, 0])
        Gsb = Gsa @ Gab
        Gsd = Gsb @ Gbd
        GsB2 = Gsd @ GdB2 #formerly Gsf


        #---------------left side---------------#

        Rac = np.array([
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1),  np.cos(theta1), 0],
            [             0,               0, 1],
        ])

        ReB1 = np.array([
            [np.cos(phi1), -np.sin(phi1), 0],
            [np.sin(phi1),  np.cos(phi1), 0],
            [           0,             0, 1],
        ])

        p_ce = np.array([0, -self.L, 0])

        Gac = SOnAndRnToSEn(Rac, [0, 0, 0])
        Gce = SOnAndRnToSEn(np.eye(3), p_ce)
        GeB1 = SOnAndRnToSEn(ReB1, [0, 0, 0])

        Gsa = SOnAndRnToSEn(np.eye(3), [x, y, 0])
        Gsc = Gsa @ Gac
        Gse = Gsc @ Gce
        GsB1 = Gse @ GeB1 #formerly Gsg

        #make objects in the frames of interest - home frame --> s frame
        line1_coords_s  = np.dot(Gsc,  self.line_coords_mat)
        line2_coords_s  = np.dot(Gsb,  self.line_coords_mat)
        box1_vertices_s = np.dot(GsB1, self.vertices_mat)
        box2_vertices_s = np.dot(GsB2, self.vertices_mat)

        #-----------#

        #convert object positions into the frame of the canvas
        box1_vert_gui    = np.dot(np.linalg.inv(self.GsGUI), box1_vertices_s)[0:2, :] 
        box2_vert_gui    = np.dot(np.linalg.inv(self.GsGUI), box2_vertices_s)[0:2, :]
        line1_coords_gui = np.dot(np.linalg.inv(self.GsGUI),  line1_coords_s)[0:2, :]
        line2_coords_gui = np.dot(np.linalg.inv(self.GsGUI),  line2_coords_s)[0:2, :]
    
        #turn line/box coords into lists of [x1, y1, x2, y2, ...]
        box1_vert_gui    = (   box1_vert_gui.T.flatten() ).astype(int)
        box2_vert_gui    = (   box2_vert_gui.T.flatten() ).astype(int)
        line1_coords_gui = (line1_coords_gui.T.flatten() ).astype(int)
        line2_coords_gui = (line2_coords_gui.T.flatten() ).astype(int)
    
        return box1_vert_gui, \
               box2_vert_gui, \
               line1_coords_gui, \
               line2_coords_gui

    #-----------------------------------#

    #event handlers

    def on_mouse_over(self, event):
        self.canvas.coords('user_posx', 
                     event.x, event.y,
                     event.x + self.coordsys_len, event.y)  
        self.canvas.coords('user_posy', 
                     event.x, event.y,
                     event.x, event.y - self.coordsys_len)
        self.mouse_posn_gui = [event.x, event.y]

        #calculate position of user in s frame
        mouse_posn_guibar = np.array([event.x, event.y, 0, 1])
        self.mouse_posn_s = np.dot(GsGUI, mouse_posn_guibar)[0:2]

    def close(self):
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

   
    def on_frame(self):
        ''' Animation update event, passed to the Tkinter canvas. Uses real-time
        data being collected and processed using the dxdt() function and the impact
        handling functions.
        '''
    
        #compare current real time to previous
        elapsed = time.perf_counter() - self.last_frametime
        elapsed_ms = int(elapsed*1000)
        prev_impact = False
    
        #elapsed time is a fraction of the total framerate in ms
        frame_delay = self.framerate_ms - elapsed_ms

        #---------------------#

        #things to be updated on each frame
        if self.q_ind < (max(self.traj_array.shape) - 1):

            #get current value of s
            t = self.t_array[self.q_ind]
            s = self.traj_array[:,self.q_ind]

            #calculate s for next timestep, and check for impact
            s_next = rk4(self.dxdt, s, t, self.dt)
            impact_dt, impact_indices  = impact_condition(s_next[0:6])

            #GUI plotting variables
            box1_vert_gui, box2_vert_gui, line1_coords_gui, line2_coords_gui = \
                self.get_GUI_coords(s)
        
            if (impact_dt):
                '''This is designed to alter the velocity of the particle
                just before impact. If we applied the impact update after impact
                (same position, changed velocity), there's a chance the objects 
                would stay stuck inside each other.
                ''' 
            
                #find phi(q) we can apply to the system. choose one to apply
                any_nearzero, phi_indices, phi_arr_np = phi_nearzero(s_next[0:6], self.atol)
                valid_phiq_indices, argmin = filter_phiq(impact_indices, phi_indices, phi_arr_np)
            
                #this is a case I eventually want to figure out
                if len(valid_phiq_indices) == 0:
                    print("Invalid phi(q)/impact condition combination") #throw an error in the future
                
                else: #valid case

                    #for fun: plot a little "spark" every time objects collide
                    body_num = (argmin//16)+1
                    vertex_ind = (argmin%16)//4
                    if body_num == 1:
                        sparks_coords = box1_vert_gui[2*vertex_ind : 2*vertex_ind + 2]
                    elif body_num == 2:
                        sparks_coords = box2_vert_gui[2*vertex_ind : 2*vertex_ind + 2]

                    self.canvas.coords('sparks', *sparks_coords)
                    make_visible(self.canvas, self.impact_photoID)
                    self.root.update_idletasks()
                    impact_eqs = impact_eqns_0_32[argmin]

                    #solve for next state, using numerical nsolve() on symbolic expressions
                    s_alt = impact_update(s, impact_eqs, self.sol_vars)
                    s_next = rk4(self.dxdt, s_alt, t, self.dt)            
         
            #apply update to trajectory vector            
            self.traj_array[:, self.q_ind+1] = s_next
            prev_impact = impact_dt
    

        #------------------GUI UPDATES------------------------#

        if prev_impact:
            make_invisible(self.canvas, self.impact_photoID)

        #apply updates to object posns
        if self.q_ind == 0:
            #create objects on the canvas
            linewidth = 2
            self.canvas.create_line(*box1_vert_gui,    tag='box1',  fill='black', width=linewidth)
            self.canvas.create_line(*box2_vert_gui,    tag='box2',  fill='black', width=linewidth)
            self.canvas.create_line(*line1_coords_gui, tag='line1', fill='blue', width=linewidth)
            self.canvas.create_line(*line2_coords_gui, tag='line2', fill='red', width=linewidth)

        else:
            #update positions of the objects by tags
            self.canvas.coords('box1', *box1_vert_gui)
            self.canvas.coords('box2', *box2_vert_gui)
            self.canvas.coords('line1', *line1_coords_gui)
            self.canvas.coords('line2', *line2_coords_gui)
    
        #see plotting_helpers.py
        label_vertices(self.canvas, box1_vert_gui, box2_vert_gui)
        self.q_ind += 1
    
        #---------------------#
    
        #update the frame delay of the timer object
        self.timer_handle = self.root.after(frame_delay, self.on_frame)
    
        #update last_frametime for next frame
        self.last_frametime = time.perf_counter()
        