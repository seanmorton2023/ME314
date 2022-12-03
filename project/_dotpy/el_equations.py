import numpy as np
import sympy as sym
import dill
from tqdm import tqdm

from geometry import *
from helpers import *


#define important positions      
ym1 = ( GsB1    @ sym.Matrix([0, 0, 0, 1]) )[0]
ym2 = ( GsB2    @ sym.Matrix([0, 0, 0, 1]) )[0]
posn_top = Gsa @ sym.Matrix([0, 0, 0, 1])


#define kinetic and potential energy

VbB1 = CalculateVb6(GsB1)
VbB2 = CalculateVb6(GsB2)

# inertia_f = m * w**2 * sym.eye(3)
# inertia_g = 

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


# In[102]:


KE_B1 = 0.5 * (VbB1.T @ inertia_B1 @ VbB1)[0]
KE_B2 = 0.5 * (VbB2.T @ inertia_B2 @ VbB2)[0]

print("KE of body 1:")
display(KE_B1)

print("KE of body 2:")
display(KE_B2)

U = m*g*(ym1 + ym2)

lagrangian = KE_B1 + KE_B2 - U
lagrangian_disp = lagrangian.simplify()

print("Lagrangian:")
display(lagrangian)
print("Simplified:")
display(lagrangian_disp)


# In[103]:


#calculate forced Euler-Lagrange equations. no forces of constraint
qd = q.diff(t)
qdd = qd.diff(t)

F_mat = sym.Matrix([
    sym.symbols(r'F_x'),
    sym.symbols(r'F_y'),
    sym.symbols(r'F_\theta1'),
    sym.symbols(r'F_\theta2'),
    sym.symbols(r'F_\phi1'),
    sym.symbols(r'F_\phi2'),
])

lhs = compute_EL_lhs(lagrangian, q)
RHS = sym.zeros(len(lhs), 1)
RHS = RHS + F_mat

#do symbolic substitutions before solving to speed up computation
subs_dict = {
    L : 1,
    w : 1/3.0,
    m : 1,
    g : 9.81,
}

lhs = lhs.subs(subs_dict)
total_eq = sym.Eq(lhs, RHS)


# In[105]:


print("Euler-Lagrange equations:")
display(total_eq)
print("Variables to solve for (transposed):")
display(qdd.T)


# In[106]:


#waited on all simplify() calls until here
t0 = time.time()
total_eq_simpl = total_eq.simplify()
display(total_eq_simpl)
tf = time.time()

print(f"Elapsed: {round(tf - t0, 2)} seconds")


# In[110]:


#attempt to round near-zero values to zero. source: https://tinyurl.com/f7t8wbmw
total_eq_rounded = total_eq_simpl
for a in sym.preorder_traversal(total_eq_simpl):
    if isinstance(a, sym.Float):
        total_eq_rounded = total_eq_rounded.subs(a, round(a, 8))
        
display(total_eq_rounded)

t0 = time.time()
#soln = sym.solve(total_eq, qdd, dict = True, simplify = False, manual = True)
#soln = sym.solve(total_eq, qdd, dict = True, manual = True)
soln = sym.solve(total_eq_rounded, qdd, dict = True, simplify = False)

tf = time.time()
print(f"Elapsed: {round(tf - t0, 2)} seconds")

eqns_solved = format_solns(soln)
for eq in eqns_solved:
    display(eq)

#simplify equations one by one
eqns_new = []
for eq in tqdm(eqns_solved):
    eq_new = sym.simplify(eq)
    eqns_new.append(eq_new)

for eq in eqns_new:
    display(eq)


#pickle the output of this constrained Euler-Lagrange derivation
temp = eqns_new
pkl_filename = 'EL_simplified.dill'
dill_dump(pkl_filename, temp)

temp = 0
display(temp)
pkl_filename = 'EL_simplified.dill'
temp = dill_load(pkl_filename)

#show temp after we've loaded it
for eq in temp:
    display(eq)


# In[131]:


# display(F_mat)
# display(q)
# display(q.diff(t))
q_ext = sym.Matrix([q, q.diff(t), F_mat])
display(q_ext.T)


# In[ ]:


#lambdify the second derivative equations and construct dynamics function
xdd_sy      = eqns_new[0]
ydd_sy      = eqns_new[1]
theta1dd_sy = eqns_new[2]
theta2dd_sy = eqns_new[3]
phi1_sy     = eqns_new[4]
phi2_sy     = eqns_new[5]

xdd_np      = sym.lambdify(q_ext,      xdd_sy)
ydd_np      = sym.lambdify(q_ext,      ydd_sy)
theta1dd_np = sym.lambdify(q_ext, theta1dd_sy)
theta2dd_np = sym.lambdify(q_ext, theta2dd_sy)
phi1dd_np   = sym.lambdify(q_ext,   phi1dd_sy)
phi2dd_np   = sym.lambdify(q_ext,   phi2dd_sy)

def dxdt(t,s):
    F_x      = 0 
    F_y      = 0
    F_theta1 = 0
    F_theta2 = 0
    F_phi1   = 0
    F_phi2   = 0
    F_array = [F_x, F_y, F_theta1, F_theta2, F_phi1, F_phi2]
    
    s_ext = np.append(s, F_array)
    #format of s: 
    #0-5: state values
    #6-11: values of derivative of state
    
    return np.array([
        *s[6:12],         
        xdd_np(*s_ext),      
        ydd_np(*s_ext),      
        theta1dd_np(*s_ext), 
        theta2dd_np(*s_ext), 
        phi1dd_np(*s_ext),   
        phi2dd_np(*s_ext),
    ])
