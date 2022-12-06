import sympy as sym
import numpy as np
import pandas as pd

import dill
import time
from tqdm import tqdm
import tkinter as tk


#from el_equations import *
#from impacts import *

from GUI import GUI
from geometry import *
from helpers import *
from plotting_helpers import *

#framerate_ms = 20 #50fps; round number preferred
#framerate_ms = 20
framerate_ms = 250


#define our sample trajectories for each angle
dt = 0.01
t_array = np.arange(0, 10, dt)
q_array_test = np.array([
    np.zeros(len(t_array)),
    0.5*(np.sin(2 * np.pi * t_array) + 1),
    (  np.pi/16) * (1 - np.sin(2 * np.pi * t_array))**2,
    -((np.pi/16) * (1 - np.sin(2 * np.pi * t_array))**2),
    2 * np.pi * t_array,
    -2 * np.pi * t_array
]).T

#q_array = q_array_test[:]
#q_array = pd.read_csv('../csv/q_array.csv', header=None).to_numpy()
q_array = pd.read_csv('../csv/q_array_impacts.csv', header=None).to_numpy()


#----------------initialize GUI----------------------#
gui = GUI(win_height, win_width) #namespace for variables: geometry.py
gui.load_arrays(q_array, line_coords_mat, vertices_mat)   #geometry.py as well
gui.load_gui_params(L_num, w_num, coordsys_len, GsGUI, framerate_ms)  

#----------------put things on the canvas----------------------#

#root and canvas defined in event handlers file
s_frame       = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')
make_grid(                    gui.canvas, win_width, win_height, pixels_to_unit)
user_coordsys = make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='user_pos')
s_frame =       make_coordsys(gui.canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')

#-------------------prep the canvas and display it------------#

gui.canvas.bind("<Button>",           gui.on_mouse_click)
gui.canvas.bind("<Motion>",           gui.on_mouse_over)
gui.root.protocol('WM_DELETE_WINDOW', gui.close) #forces closing of all Tk() functions

print(gui.GsGUI)
print(np.linalg.inv(gui.GsGUI))
gui.canvas.pack()
gui.timer_id = gui.root.after(gui.framerate_ms, gui.on_timer)

gui.root.mainloop()



