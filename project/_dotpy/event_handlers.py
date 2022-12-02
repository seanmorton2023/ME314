import tkinter as tk

def on_mouse_click(event):
    print(f"Mouse clicked at: {event.x} {event.y}")

def on_mouse_over(event):
    #canvas.coords(line, 0, 0, event.x, event.y)
    canvas.coords('user_posx', 
                 event.x, event.y,
                 event.x + coordsys_len, event.y
             )  
    
    canvas.coords('user_posy', 
                 event.x, event.y,
                 event.x, event.y - coordsys_len
             )
    
def on_key(event):
    print(event.char)
    print(event.keycode)
    if event.char.lower() == 'q':
        root.destroy()

def on_frame(event):
    pass

def close():
    try:
        root.quit()
        root.destroy()
    except:
        pass
    
def on_timer():
    '''
    Animation update event, passed to the Tkinter canvas. Uses real-time
    data so that framerate is consistent.
    '''
    #arguments will be global variables to pass function to Tk
    global framerate_ms, q_array
    global L, w, line_coords_mat, vertices_mat
    
    # global variables we expect to change
    global timer_handle, last_frametime, q_ind    
    
    #compare current real time to previous
    elapsed = time.perf_counter() - last_frametime
    elapsed_ms = int(elapsed*1000)
    
    #elapsed time is a fraction of the total framerate in ms
    frame_delay = framerate_ms - elapsed_ms
    
    #---------------------#
    
    #things to be updated on each frame
    if q_ind < len(q_array):
        q = q_array[q_ind]

        #see function above for how we alter the coords to get them in GUI frame
        box1_vert_gui, box2_vert_gui, line1_coords_gui, line2_coords_gui = \
            get_GUI_coords(q, line_coords_mat, vertices_mat, GsGUI, L, w)
    
    #apply updates to object posns
    if q_ind == 0:
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
    
    #update index of observation for q
    q_ind += 1
    
    #---------------------#

    print("\nGUI debug:")
    print(f"q: \n{q}")
    print(f"box1_vert_gui: \n{box1_vert_gui}")
    
    #update the frame delay of the timer object
    timer_id = root.after(frame_delay, on_timer)
    
    #update last_frametime for next frame
    last_frametime = time.perf_counter()

canvas.bind("<Button>", on_mouse_click)
canvas.bind("<Motion>", on_mouse_over)
canvas.bind("<Key>", on_key)
root.protocol('WM_DELETE_WINDOW', close) #forces closing of all Tk() functions

