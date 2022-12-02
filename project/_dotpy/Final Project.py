import sympy as sym
import numpy as np
import dill
import time
from tqdm import tqdm
import tkinter as tk

from el_equations import *
from event_handlers import *
from geometry import *
from get_GUI_coords import *
from helpers import *
from impacts import *
from plotting_helpers import *

root = tk.Tk()
root.title("Final Project")
win_height = 600
win_width = 800
canvas = tk.Canvas(root, bg="white", height=win_height, width=win_width)

#line = canvas.create_line(0, 0, 400, 400)
pixels_to_unit = 100
coordsys_len = 50
user_coordsys = make_coordsys(canvas, 300, 400, coordsys_len, tag='user_pos')
s_frame = make_coordsys(canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')
make_grid(canvas, win_width, win_height, pixels_to_unit)

#define our sample trajectories for each angle
dt = 0.01
t_array = np.arange(0, 10, dt)
q_array_test = np.array([
    np.zeros(len(t_array)),
    np.sin(2 * np.pi * t_array) + 1,
    (np.pi/8) * (1 - (np.sin(2 * np.pi * t_array))**2),
    -((np.pi/8) * (1 - np.sin(2 * np.pi * t_array))**2),
    2 * np.pi * t_array,
    -2 * np.pi * t_array
]).T

q_array = q_array_test[:]

#generate the GUI controls
root = tk.Tk()
root.title("Final Project")
win_height = 600
win_width = 800
canvas = tk.Canvas(root, bg="white", height=win_height, width=win_width)

pixels_to_unit = 100
coordsys_len = 50
s_frame = make_coordsys(canvas, win_width/2, win_height/2, coordsys_len, tag='s_frame')
make_grid(canvas, win_width, win_height, pixels_to_unit)

#framerate_ms = 20 #50fps; round number preferred
framerate_ms = 5000 #for debug
frame_delay_init = framerate_ms
last_frametime = time.perf_counter()
q_ind = 0    


# In[93]:


#have these variables in global namespace
L = 1
w = 1/3.0
width  = win_width  // pixels_to_unit
height = win_height // pixels_to_unit

#let frame GUI be the coordinates as seen on the GUI,
#frame r be the frame at GUI coords (0,0) with axes in same direction
#as frame s
GrGUI = np.array([
    [width/win_width,    0, 0, 0],
    [0, -height/win_height, 0, 0],
    [0,                      0, 1, 0],
    [0,                      0, 0, 1]
]) 

Grs = SOnAndRnToSEn(np.identity(3), [width/2, -height/2, 0])
GsGUI = np.dot(InvSEn(Grs), GrGUI)
display(GsGUI)
display(np.linalg.inv(GsGUI))
canvas.pack()
timer_id = root.after(framerate_ms,on_timer)

root.mainloop()



