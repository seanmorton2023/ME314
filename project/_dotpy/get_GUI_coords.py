import numpy as np
import sympy as sym
import tkinter as tk


def get_GUI_coords(q, line_coords_mat, vertices_mat, GsGUI, L, w):
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

    p_bd = np.array([0, -L, 0])

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
        [              0,                0, 1],
    ])

    ReB1 = np.array([
        [np.cos(phi1), -np.sin(phi1), 0],
        [np.sin(phi1),  np.cos(phi1), 0],
        [            0,              0, 1],
    ])

    p_ce = np.array([0, -L, 0])

    Gac = SOnAndRnToSEn(Rac, [0, 0, 0])
    Gce = SOnAndRnToSEn(np.eye(3), p_ce)
    GeB1 = SOnAndRnToSEn(ReB1, [0, 0, 0])

    Gsa = SOnAndRnToSEn(np.eye(3), [x, y, 0])
    Gsc = Gsa @ Gac
    Gse = Gsc @ Gce
    GsB1 = Gse @ GeB1 #formerly Gsg

    #make objects in the frames of interest - 

    #line L1, frame C -> s frame
    line1_coords_s = np.dot(Gsc, line_coords_mat)

    #line L2, frame B -> s frame
    line2_coords_s = np.dot(Gsb, line_coords_mat)

    #box 1, frame B1 -> s frame
    box1_vertices_s = np.dot(GsB1, vertices_mat)

    #box 2, frame B2 -> s frame
    box2_vertices_s = np.dot(GsB2, vertices_mat)


    #-----------#

    #convert object positions into the frame of the canvas
    box1_vert_gui    = np.dot(np.linalg.inv(GsGUI), box1_vertices_s)[0:2, :] 
    box2_vert_gui    = np.dot(np.linalg.inv(GsGUI), box2_vertices_s)[0:2, :]
    line1_coords_gui = np.dot(np.linalg.inv(GsGUI),  line1_coords_s)[0:2, :]
    line2_coords_gui = np.dot(np.linalg.inv(GsGUI),  line2_coords_s)[0:2, :]
    
    #turn line/box coords into lists of [x1, y1, x2, y2, ...]
    box1_vert_gui    = (   box1_vert_gui.T.flatten() ).astype(int)
    box2_vert_gui    = (   box2_vert_gui.T.flatten() ).astype(int)
    line1_coords_gui = (line1_coords_gui.T.flatten() ).astype(int)
    line2_coords_gui = (line2_coords_gui.T.flatten() ).astype(int)
    
    return box1_vert_gui, \
           box2_vert_gui, \
           line1_coords_gui, \
           line2_coords_gui