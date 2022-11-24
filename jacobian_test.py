#define variables
import sympy as sym
import numpy as np

def SOnAndRnToSEn(R, p):
       
#     print('-----------------------')
#     print("\nSOnAndRn... Debug:")
#     print("\nR:")
#     print(type(R))
#     print(R)
#     print("\np:")
#     print(type(p))
#     print(p)
    
    #do type checking for the matrix types
    if type(R) == list:
        R = np.matrix(R)
        
    n = R.shape[0]
    if ((R.shape[0] != R.shape[1]) or                               #R is NP array or Sym matrix
        ((type(p) is np.ndarray and max(p.shape) != R.shape[0]) or  #p is NP array and shape mismatch or.. 
          ((isinstance(p, list) or isinstance(p, sym.Matrix)) and 
            ( len(p) != R.shape[0] ))   )  ):                       #p is Sym matrix or "list" and shape mismatch
        raise Exception(f"Shape of R {R.shape} and p ({len(p)}) mismatch; exiting.")
        return None
        
    #construct a matrix based on returning a Sympy Matrix
    if isinstance(R, sym.Matrix) or isinstance(p, sym.Matrix): 
        #realistically one of these needs to be symbolic to do this

        if isinstance(R, np.ndarray) or isinstance(p, np.ndarray):
            raise Exception("R and p cannot mix/match Sympy and Numpy types")
            return None
        
        G = sym.zeros(n+1)
        G[:n, n] = sym.Matrix(p)
    
    #construct a matrix based on returning a Numpy matrix
    elif isinstance(R, np.ndarray) or isinstance(R, list):
        G = np.zeros([n+1, n+1])
        # print(f"\nSOnAndRnToSEn Debug: \n\nR:\n{R}    \n\np:\n{p}   ")
        G[:n, n] = np.array(p).T
        
    else:
        raise Exception("Error: type not recognized")
        return None
    
    G[:n,:n] = R
    G[-1,-1] = 1
    return G  



L1, L2, m, J, W, g = sym.symbols(r'L_1, L_2, m, J, W, g')
t = sym.symbols(r't')
x = sym.Function(r'x')(t)
y = sym.Function(r'y')(t)
theta1 = sym.Function(r'\theta_1')(t)
theta2 = sym.Function(r'\theta_2')(t)

q = sym.Matrix([x, y, theta1, theta2])
qd = q.diff(t)
qdd = qd.diff(t)

#not included: xd, yd, ..., q_ext

#define transformation matrices. let A1 be in the direction of 
#the right leg and A2 be in the direction of the left leg

#------right leg------#
Raa1 = sym.Matrix([
    [sym.cos(theta1), -sym.sin(theta1), 0],
    [sym.sin(theta1),  sym.cos(theta1), 0],
    [              0,                0, 1]
])

Rdf = sym.Matrix([
    [sym.cos(-theta1), -sym.sin(-theta1), 0],
    [sym.sin(-theta1),  sym.cos(-theta1), 0],
    [              0,                0,   1]
])

p_a1b = sym.Matrix([0, -L1/2, 0])
p_bd =  sym.Matrix([0, -L1/2, 0])

Gaa1 = SOnAndRnToSEn(Raa1, [0,0,0]) 
