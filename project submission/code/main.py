import sympy as sym
import numpy as np
import pandas as pd

import dill
import time
from tqdm import tqdm
import tkinter as tk

from GUI import GUI
from geometry import *
from helpers import *
from plotting_helpers import *
from el_equations import *
from impacts import *

#-----------------tuning parameters-----------------------------#

#time parameters
framerate_ms = 20
dt = 0.005
t_span = [0, 5]
t_array = np.arange(t_span[0], t_span[1], dt)

theta0 = np.pi/4
init_posns = [0, 1, theta0, -theta0, np.pi/2, np.pi/4]
init_velocities = [0, 0, -1, 1, 7, -4]
ICs = init_posns + init_velocities

#spring and damping constants for PD control
k = 30
Bx = 2
By = 5

#tolerance for detecting if phi(q) near zero
atol = 1E-1

#define trajectory for particle to follow
y_tracking = lambda t: -np.sin(3*np.pi*t)+1
x_tracking = lambda t: 0

#forces use PD control in x and y to track a position
F_eqs_array = np.array([
    lambda s,t: k*(x_tracking(t) - s[0]) - Bx*s[6] , #F_x - damping term applied to vel.
    lambda s,t: k*(y_tracking(t) - s[1]) - By*s[7] + 19.62, #F_y
    lambda s,t: 0, #F_theta1
    lambda s,t: 0, #F_theta2
    lambda s,t: 0, #F_phi1
    lambda s,t: 0, #F_phi2
])

dxdt = construct_dxdt(F_eqs_array)


#----------------initialize GUI----------------------#

gui = GUI(win_height, win_width) #namespace for variables: geometry.py
gui.load_arrays(line_coords_mat, vertices_mat)   #geometry.py as well
gui.load_gui_params(L_num, w_num, coordsys_len, GsGUI, 
                    framerate_ms, '../sprites/impact_sparks.png') #plotting_helpers.py
gui.load_simulation(dxdt, t_span, dt, ICs, atol)

#----------------populate canvas----------------------#

s_frame       = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')
make_grid(                    gui.canvas, win_width,   win_height,   pixels_to_unit)
user_coordsys = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='user_pos')
s_frame =       make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')

#--------------canvas display--------------------#

gui.canvas.bind("<Motion>",           gui.on_mouse_over)
gui.root.protocol('WM_DELETE_WINDOW', gui.close) #forces closing of all Tk() functions
gui.canvas.pack()

gui.timer_id = gui.root.after(gui.framerate_ms, gui.on_frame)
gui.root.mainloop()

