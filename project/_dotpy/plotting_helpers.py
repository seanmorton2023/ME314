#helper functions for GUI operations
import tkinter as tk
import time
import numpy as np

from helpers import *
from geometry import *

def make_oval(canvas: tk.Canvas, center: tuple, width: int, height: int, fill: str='hotpink'):
    top_left = (center[0] - width, center[1] - height)
    bottom_right = (center[0] + width, center[1] + height)
    return canvas.create_oval([top_left, bottom_right], fill=fill, width=0) #return content ID

def make_circle(canvas: tk.Canvas, center: tuple, radius: int, fill: str='hotpink'):
    return make_oval(canvas, center, radius, radius, fill=fill) #return content ID

def make_grid_label(canvas, x, y, w, h, offset, pixels_to_unit):
        
    #apply offset by finding origin and applying conversion
    #from pixels to units in world
    width_world = w//pixels_to_unit
    height_world = h//pixels_to_unit
    
    origin_x = width_world//2
    origin_y = height_world//2
        
    xlabel, ylabel = (x/pixels_to_unit - origin_x, (h-y)/pixels_to_unit - origin_y - 0.5)

    #decide whether label is for x or y
    coord = xlabel if not xlabel == -origin_x else ylabel
    
    canvas.create_oval(
        x - offset, 
        y - offset, 
        x + offset,  
        y + offset, 
        fill='black'
    )
    canvas.create_text(
        x + offset, 
        y + offset, 
        text=str(round(coord,1)),
        anchor="sw", 
        font=("Purisa", 12)
    )

def make_grid(canvas, w, h, interval):
    #interval = the # of pixels per unit distance in the simulation
    
    # Delete old grid if it exists:
    canvas.delete('grid_line')
    offset = 2

    # Creates all vertical lines every 0.5 unit
    #for i in range(0, w, interval):
    for i in np.linspace(0, w, 2*w//interval+1).tolist()[:-1]:
        canvas.create_line(i, 0, i, h, tag='grid_line', fill='gray', dash=(2,2))
        make_grid_label(canvas, i, h, w, h, offset, interval)

    # Creates all horizontal lines every 0.5 unit
    #for i in range(0, h, interval):
    for i in np.linspace(0, h, 2*h//interval+1).tolist()[:-1]:
        canvas.create_line(0, i, w, i, tag='grid_line', fill='gray', dash=(2,2))
        make_grid_label(canvas, 0, i, w, h, offset, interval)

        
def make_coordsys(canvas, x, y, line_length, tag):
    canvas.create_line(x, y, x + line_length,               y, arrow=tk.LAST, tag=tag+'x')
    canvas.create_line(x, y,               x, y - line_length, arrow=tk.LAST, tag=tag+'y')

def label_vertices(canvas, box1_vert_gui, box2_vert_gui):
    '''For debug purposes, put labels on each vertex of the boxes so we can see
    which impact conditions are occurring at a given point in time.

    Box1_vert_gui and box2_vert_gui are 10x0 flattened arrays, (x1, y1, x2, y2,...)
    '''

    #remove 5th set of box vertices, as it closes the box structure
    box1_vert_gui = np.array(box1_vert_gui)[:-2]
    box2_vert_gui = np.array(box2_vert_gui)[:-2]

    canvas.delete("Vertices")
    offset = 2

    for i in range(len(box1_vert_gui)//2):
        x, y = box1_vert_gui[2*i : 2*i + 2]
        canvas.create_text(
            x + offset, 
            y - offset, 
            text=f"V1{i+1}",
            anchor="s", 
            font=("Purisa", 8),
            tag="Vertices"
        )

    for i in range(len(box2_vert_gui)//2):
        x, y = box2_vert_gui[2*i : 2*i + 2]
        canvas.create_text(
            x + offset, 
            y - offset, 
            text=f"V2{i+1}",
            anchor="s", 
            font=("Purisa", 8),
            tag="Vertices"
        )








