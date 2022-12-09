import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

from helpers import *

#define frames and symbols. let L1 = L2, m1 = m2 for computational efficiency
#as this condition is unlikely to change
L, w, m, g = sym.symbols(r'L, w, m, g')
t = sym.symbols(r't')

x = sym.Function(r'x')(t)
y = sym.Function(r'y')(t)
theta1 = sym.Function(r'\theta_1')(t)
theta2 = sym.Function(r'\theta_2')(t)
phi1 = sym.Function(r'\Phi_1')(t)
phi2 = sym.Function(r'\Phi_2')(t)

q = sym.Matrix([x, y, theta1, theta2, phi1, phi2])
q_ext = sym.Matrix([q, q.diff(t)])

subs_dict = {
    L : 1,
    w : 1/6.0,
    m : 1,
    g : 9.81,
}

#make sure geometry for plotting matches 
#the Sympy substitution dict!
L_num = 1
w_num = 1/6.0

#--------symbolic transformation matrices, right side---------------#

Rab = sym.Matrix([
    [sym.cos(theta2), -sym.sin(theta2), 0],
    [sym.sin(theta2),  sym.cos(theta2), 0],
    [              0,                0, 1],
])

RdB2 = sym.Matrix([
    [sym.cos(phi2), -sym.sin(phi2), 0],
    [sym.sin(phi2),  sym.cos(phi2), 0],
    [            0,              0, 1],
])

p_bd = sym.Matrix([0, -L, 0])

Gab = SOnAndRnToSEn(Rab, [0, 0, 0])
Gbd = SOnAndRnToSEn(sym.eye(3), p_bd)
GdB2 = SOnAndRnToSEn(RdB2, [0, 0, 0])

Gsa = SOnAndRnToSEn(sym.eye(3), [x, y, 0])
Gsb = Gsa @ Gab
Gsd = Gsb @ Gbd
GsB2 = Gsd @ GdB2 #formerly Gsf


#--------symbolic transformation matrices, left side---------------#

Rac = sym.Matrix([
    [sym.cos(theta1), -sym.sin(theta1), 0],
    [sym.sin(theta1),  sym.cos(theta1), 0],
    [              0,                0, 1],
])

ReB1 = sym.Matrix([
    [sym.cos(phi1), -sym.sin(phi1), 0],
    [sym.sin(phi1),  sym.cos(phi1), 0],
    [            0,              0, 1],
])

p_ce = sym.Matrix([0, -L, 0])

Gac = SOnAndRnToSEn(Rac, [0, 0, 0])
Gce = SOnAndRnToSEn(sym.eye(3), p_ce)
GeB1 = SOnAndRnToSEn(ReB1, [0, 0, 0])

Gsa = SOnAndRnToSEn(sym.eye(3), [x, y, 0])
Gsc = Gsa @ Gac
Gse = Gsc @ Gce
GsB1 = Gse @ GeB1 #formerly Gsg


#------------line + box geometry; plotting geometry-------------------#

#Lnum and wnum defined under subs_dict

win_height = 600
win_width = 800
pixels_to_unit = 200
coordsys_len = 50

vertices_mat = np.array([
    [ w_num/2.0,  w_num/2.0, 0, 1], 
    [-w_num/2.0,  w_num/2.0, 0, 1],
    [-w_num/2.0, -w_num/2.0, 0, 1],
    [ w_num/2.0, -w_num/2.0, 0, 1],
    [ w_num/2.0,  w_num/2.0, 0, 1], #add first vertex onto end of matrix again so line wraps around
]).T #in "bar" form  so they can be multiplied by trans. matrix

line_coords_mat = np.array([
    [0,     0, 0, 1],
    [0, -L_num, 0, 1],
]).T

#have these variables in global namespace
width  = win_width  // pixels_to_unit
height = win_height // pixels_to_unit

#let frame GUI be the coordinates as seen on the GUI,
#frame r be the frame at GUI coords (0,0) with axes in same direction
#as frame s. This is not in SE(3) so InvSEn() cannot be used with this.
GrGUI = np.array([
    [width/win_width,    0, 0, 0],
    [0, -height/win_height, 0, 0],
    [0,                  0, 1, 0],
    [0,                  0, 0, 1]
]) 

Grs = SOnAndRnToSEn(np.identity(3), [width/2, -height/2, 0])
GsGUI = np.dot(InvSEn(Grs), GrGUI)

#define important positions      
ym1 = ( GsB1    @ sym.Matrix([0, 0, 0, 1]) )[1]
ym2 = ( GsB2    @ sym.Matrix([0, 0, 0, 1]) )[1]
posn_top = Gsa @ sym.Matrix([0, 0, 0, 1])


#----------------------------------------------#

#inertial properties of system, in symbolic form.
#reused in both Lagrangian and Hamiltonian calculation,
#so define it once

VbB1 = CalculateVb6(GsB1,t)
VbB2 = CalculateVb6(GsB2,t)

scriptI_B1 = m * sym.Matrix([
    [w**2,    0,      0],
    [   0, w**2,      0],
    [   0,    0, 2*w**2]
])

scriptI_B2 = m * sym.Matrix([
    [w**2,    0,      0],
    [   0, w**2,      0],
    [   0,    0, 2*w**2]
])

inertia_B1 = InertiaMatrix6(m, scriptI_B1)
inertia_B2 = InertiaMatrix6(m, scriptI_B2)

