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


#framerate_ms = 20 #50fps; round number preferred
#framerate_ms = 100
framerate_ms = 20

#define our sample trajectories for each angle
dt = 0.005
t_span = [0, 5]
t_array = np.arange(t_span[0], t_span[1], dt)

#define trajectory for particle to follow
#y_tracking = lambda t: np.cos(3*np.pi*t)
#y_tracking = lambda t: np.cos(2*1.8*np.pi*t)
#y_tracking = lambda t: 0.6*(np.cos(3*np.pi*t)+1)
#y_tracking = lambda t: 0.7*(-np.sin(3*np.pi*t)+1)
y_tracking = lambda t: -np.sin(3*np.pi*t)+1

x_tracking = lambda t: 0

#define spring and damping values
#k = 20
k = 30
Bx = 2
By = 5

impact_eqns_0_32 = dill_load('../dill/impact_eqns_0_32.dill')
atol = 1E-1


F_eqs_array = np.array([
    #lambda s,t: 0, #F_x
    lambda s,t: k*(x_tracking(t) - s[0]) - Bx*s[6] , #F_x - damping term applied to vel.
    lambda s,t: k*(y_tracking(t) - s[1]) - By*s[7] + 19.62, #F_y
    lambda s,t: 0, #F_theta1
    lambda s,t: 0, #F_theta2
    lambda s,t: 0, #F_phi1
    lambda s,t: 0, #F_phi2
])

#F_eqs_array = np.array([
#    #lambda s,t: 0, #F_x
#    lambda s,t: k*(x_tracking(t) - s[0]), #F_x - damping term applied to vel.
#    lambda s,t: k*(y_tracking(t) - s[1]) + 19.62, #F_y
#    lambda s,t: 0, #F_theta1
#    lambda s,t: 0, #F_theta2
#    lambda s,t: 0, #F_phi1
#    lambda s,t: 0, #F_phi2
#])

dxdt = construct_dxdt(F_eqs_array)


#theta0 = np.pi/2 - 0.2
theta0 = np.pi/4

init_posns = [0, 1, theta0, -theta0, np.pi/2, np.pi/4]
init_velocities = [0, 0, -1, 1, 7, -4]
#init_velocities = [0, 0, -1, 1, 0, 0]

ICs = init_posns + init_velocities


q_array_test = np.array([
    np.zeros(len(t_array)),
    0.5*(np.sin(2 * np.pi * t_array) + 1),
    (  np.pi/16) * (1 - np.sin(2 * np.pi * t_array))**2,
    -((np.pi/16) * (1 - np.sin(2 * np.pi * t_array))**2),
    2 * np.pi * t_array,
    -2 * np.pi * t_array
]).T

q_array = q_array_test[:]
#q_array = pd.read_csv('../csv/q_array.csv', header=None).to_numpy()
#q_array = pd.read_csv('../csv/q_array_impacts.csv', header=None).to_numpy()


#----------------initialize GUI----------------------#
gui = GUI(win_height, win_width) #namespace for variables: geometry.py
gui.load_arrays(q_array, line_coords_mat, vertices_mat)   #geometry.py as well
gui.load_gui_params(L_num, w_num, coordsys_len, GsGUI, 
                    framerate_ms, '../sprites/impact_sparks.png') #plotting_helpers.py
gui.load_simulation(dxdt, t_span, dt, ICs, atol)

#----------------put things on the canvas----------------------#

#root and canvas defined in event handlers file
s_frame       = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')
make_grid(                    gui.canvas, win_width,   win_height,   pixels_to_unit)
user_coordsys = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='user_pos')
s_frame =       make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')

#-------------------prep the canvas and display it------------#

gui.canvas.bind("<Button>",           gui.on_mouse_click)
gui.canvas.bind("<Motion>",           gui.on_mouse_over)
gui.root.protocol('WM_DELETE_WINDOW', gui.close) #forces closing of all Tk() functions

print(gui.GsGUI)
print(np.linalg.inv(gui.GsGUI))
gui.canvas.pack()
#gui.timer_id = gui.root.after(gui.framerate_ms, gui.on_timer)
gui.timer_id = gui.root.after(gui.framerate_ms, gui.on_frame_v2)


gui.root.mainloop()


    ##save results to a CSV file for use in animation. This needs to be removed before submission.
    #pd.DataFrame(traj_array.T).to_csv('../csv/q_array_impacts.csv', header=None, index=None)
    #return traj_array
