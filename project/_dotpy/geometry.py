import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

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
    w : 1/3.0,
    m : 1,
    g : 9.81,
}

#---------------right side---------------#

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


#---------------left side---------------#

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

#-------------------------------------#

#geometry for plotting

vertices_mat = w * np.array([
    [ np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1],
    [-np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1],
    [-np.sqrt(2)/2, -np.sqrt(2)/2, 0, 1],
    [ np.sqrt(2)/2, -np.sqrt(2)/2, 0, 1],
    [ np.sqrt(2)/2,  np.sqrt(2)/2, 0, 1], #add first vertex onto end of matrix again so line wraps around
]).T #in "bar" form  so they can be multiplied by trans. matrix

line_coords_mat = np.array([
    [0,  0, 0, 1],
    [0, -L, 0, 1],
]).T