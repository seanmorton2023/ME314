#helper functions for GUI operations
import tkinter as tk
import time
import numpy as np

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
        
    xlabel, ylabel = (x//pixels_to_unit - origin_x, (h-y)//pixels_to_unit - origin_y)

    #decide whether label is for x or y
    coord = xlabel if not x==0 else ylabel
    
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
        text="{0}".format(coord),
        anchor="sw", 
        font=("Purisa", 12)
    )

def make_grid(canvas, w, h, interval):
    #interval = the # of pixels per unit distance in the simulation
    
    # Delete old grid if it exists:
    canvas.delete('grid_line')
    # Creates all vertical lines at intervals of 100
    for i in range(0, w, interval):
        canvas.create_line(i, 0, i, h, tag='grid_line', fill='gray', dash=(2,2))

    # Creates all horizontal lines at intervals of 100
    for i in range(0, h, interval):
        canvas.create_line(0, i, w, i, tag='grid_line', fill='gray', dash=(2,2))

    # Creates axis labels
    offset = 2
    for y in range(0, h, interval):
        make_grid_label(canvas, 0, y, w, h, offset, interval)
        
    for x in range(0, w, interval):
        make_grid_label(canvas, x, h, w, h, offset, interval)
        
def make_coordsys(canvas, x, y, line_length, tag):
    canvas.create_line(x, y, x + line_length,               y, arrow=tk.LAST, tag=tag+'x')
    canvas.create_line(x, y,               x, y - line_length, arrow=tk.LAST, tag=tag+'y')


