import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

from geometry import *

#-----------GEOMETRIC FUNCTIONS-----------#

def SOnAndRnToSEn(R, p):
           
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

def SEnToSOnAndRn(SEnmat):
    '''Decomposes a SE(n) vector into its rotation matrix and displacement components.
    '''
    if isinstance(SEnmat, list):
        SEnmat = np.matrix(SEnmat)
    n = SEnmat.shape[0]
    return SEnmat[:(n-1), :(n-1)], SEnmat[:(n-1), n-1]

def HatVector3(w):
    '''Turns a vector in R3 to a skew-symmetric matrix in so(3). 
    Works with both Sympy and Numpy matrices.
    '''   
    #create different datatype representations based on type of w
    if isinstance(w, list) or isinstance(w, np.ndarray) \
        or isinstance(w, np.matrix):
        f = np.array 
    elif isinstance(w, sym.Matrix): #NP and Sym
        f = sym.Matrix

    return f([
        [    0, -w[2],  w[1]],
        [ w[2],     0, -w[0]],
        [-w[1],  w[0],     0]
    ])

def UnhatMatrix3(w_hat):
    '''Turns a skew-symmetric matrix in so(3) into a vector in R3.
    '''
    if isinstance(w_hat, list) or isinstance(w_hat, np.ndarray) \
        or isinstance(w_hat, np.matrix):
        f = np.array
        w_hat = np.array(w_hat)
    elif isinstance(w_hat, sym.Matrix) or isinstance(w_hat, sym.ImmutableMatrix):
        f = sym.Matrix
    else:
        raise Exception(f"UnhatMatrix3: Unexpected type of w_hat: {type(w_hat)}")
    
    #matrix checking, for use in potential debug. generalized to both Sympy and Numpy
    same = np.array([w_hat + w_hat.T == f([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    ) ] )
    
#     if (not same.all()):
#         raise Exception("UnhatMatrix3: w_hat not skew_symmetric")
    
    #NP and Sym
    return f([
        -w_hat[1,2],
        w_hat[0,2],
        -w_hat[0,1],
    ])

def InvSEn(SEnmat):
    '''Takes the inverse of a SE(n) matrix.
    Compatible with Numpy, Sympy, and list formats.
    '''
    if isinstance(SEnmat, list):
        SEnmat = np.matrix(SEnmat)
    ###
    n = SEnmat.shape[0]
    R = SEnmat[:(n-1), :(n-1)]
    p = SEnmat[:(n-1),   n-1 ]
        
    return SOnAndRnToSEn(R.T, -R.T @ p)
    
def InertiaMatrix6(m, scriptI):
    '''Takes the mass and inertia matrix properties of an object in space,
    and constructs a 6x6 matrix corresponding to [[mI 0]; [0 scriptI]].
    Currently only written for Sympy matrix representations.
    '''
    if (m.is_Matrix or not scriptI.is_square):
        raise Exception("Type error: m or scriptI in InertiaMatrix6")
        
    mat = sym.zeros(6)
    mI = m * sym.eye(3)
    mat[:3, :3] = mI
    mat[3:6, 3:6] = scriptI
    return mat

def HatVector6(vec):
    '''Convert a 6-dimensional body velocity into a 4x4 "hatted" matrix,
    [[w_hat v]; [0 0]], where w_hat is skew-symmetric.
    w = 
    '''
    if isinstance(vec, np.matrix) or isinstance(vec, np.ndarray):
        vec = np.array(vec).flatten()
    
    v = vec[:3]
    w = vec[3:6]
    
    #this ensures if there are symbolic variables, they stay in Sympy form
    if isinstance(vec, sym.Matrix):
        v = sym.Matrix(v)
        w = sym.Matrix(w)
        
    w_hat = HatVector3(w)
    
    #note that the result isn't actually in SE(3) but 
    #that the function below creates a 4x4 matrix from a 3x3 and
    #1x3 matrix - with type checking - so we'll use it
    mat = SOnAndRnToSEn(w_hat, v)
    return mat

def UnhatMatrix4(mat):
    '''Convert a 4x4 "hatted" matrix,[[w_hat v]; [0 0]], into a 6-dimensional
    body velocity [v, w].
    '''
    #same as above - matrices aren't SE(3) and SO(3) but the function
    #can take in a 4x4 mat and return a 3x3 and 3x1 mat
    [w_hat, v] = SEnToSOnAndRn(mat)
    w = UnhatMatrix3(w_hat)
    
    if (isinstance(w, np.matrix) or isinstance(w, np.ndarray)):
        return np.array([v, w]).flatten()
    elif isinstance(w, sym.Matrix):
        return sym.Matrix([v, w])
    else:
        raise Exception("Unexpected datatype in UnhatMatrix4")
    
def CalculateVb6(G,t):
    '''Calculate the body velocity, a 6D vector [v, w], given a trans-
    formation matrix G from one frame to another.
    '''
    G_inv = InvSEn(G)
    Gdot = G.diff(t) #for sympy matrices, this also carries out chain rule 
    V_hat = G_inv @ Gdot 
    
#     if isinstance(G, sym.Matrix):
#         V_hat = sym.simplify(V_hat)
        
    return UnhatMatrix4(V_hat)

#-----------EULER-LAGRANGE -----------#

def compute_EL_lhs(lagrangian, q, t):
    '''
    Helper function for computing the Euler-Lagrange equations for a given system,
    so I don't have to keep writing it out over and over again.
    
    Inputs:
    - lagrangian: our Lagrangian function in symbolic (Sympy) form
    - q: our state vector [x1, x2, ...], in symbolic (Sympy) form
    
    Outputs:
    - eqn: the Euler-Lagrange equations in Sympy form
    '''
    
    # wrap system states into one vector (in SymPy would be Matrix)
    #q = sym.Matrix([x1, x2])
    qd = q.diff(t)
    qdd = qd.diff(t)

    # compute derivative wrt a vector, method 1
    # wrap the expression into a SymPy Matrix
    L_mat = sym.Matrix([lagrangian])
    dL_dq = L_mat.jacobian(q)
    dL_dqdot = L_mat.jacobian(qd)

    #set up the Euler-Lagrange equations
    #LHS = dL_dq - dL_dqdot.diff(t)
    LHS = dL_dqdot.diff(t) - dL_dq
    
    return LHS.T

def format_solns(soln):
    eqns_solved = []
    #eqns_new = []

    for i, sol in enumerate(soln):
        for x in list(sol.keys()):
            eqn_solved = sym.Eq(x, sol[x])
            eqns_solved.append(eqn_solved)

    return eqns_solved

#-----------DATA SAVING FUNCTIONS-----------#


def dill_dump(filename, data):
    dill.settings['recurse'] = True
    with open(filename, 'wb') as f:
        dill.dump(data, f)
        
def dill_load(filename):
    dill.settings['recurse'] = True
    with open(filename, 'rb') as f:
        data = dill.load(f)
    return data

def write_csv_line(csv_filename, data):
    #Appends a single line of data to a CSV file.
    with open(csv_filename, 'a') as f:
        data_str = ','.join([str(i) for i in data]) + '\n'
        f.write(data_str)

def write_csv_mat(csv_filename, mat):
    #Clears out existing data in a CSV file and writes a new matrix
    #of data to the file.

    f = open(csv_filename, 'w') #clear out old data
    f.close()

    #datatype handling
    mat = np.matrix(mat).tolist()
    print("\nSaving data...")
    for row in tqdm(mat):
            write_csv_line(csv_filename, row)

#-----------TESTING FUNCTIONS-----------#

def TestDill(lagrangian):
    #test out pickling a symbolic expression using Dill
    dill.settings['recurse'] = True
    filename ='test_sym_matrix.dill'
    with open(filename, 'wb') as f:
        dill.dump(lagrangian, f)

    lagrangian = 0
    print("Value before dill load:")
    display(lagrangian)
    
    with open(filename, 'rb') as f:
        lagrangian = dill.load(f)
    
    print("Value after dill load:")
    display(lagrangian)

def TestHat3():

    #testing
    t = sym.symbols(r't')
    theta1 = sym.Function(r'\theta_1')(t)
    theta2 = sym.Function(r'\theta_2')(t)

    w1 = [6,5,4]

    w_hat1 = [
        [0, -9, 8],
        [9, 0, -7],
        [-8, 7, 0]
    ]

    w_hat2 = [
        [10, -9, 8],
        [9, 0, -7],
        [-8, 7, 0]
    ]

    w_hat3 = sym.Matrix([
        [0, -theta1, 8],
        [theta1, 0, -7],
        [-8, 7, 0]
    ])

    T1 = [
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9, 10, 11, 12],
        [0,  0,  0,  1]
    ]

    print(f"\nHat: \n {w1} \n{HatVector3(w1)}")
    print(f"\nUnhat: \n{w_hat1} \n{UnhatMatrix3(w_hat1)}")
    #print(f"\nNon-Skew-symm unhat: \n{w_hat2} \n{UnhatMatrix3(w_hat2)}")
    print(f"\nSymbolic unhat: \n{w_hat3} \n{UnhatMatrix3(w_hat3)}")

    print(f"\nTransInv: \n{T1} \n{InvSEn(T1)}")

def TestMatrix4():

    ### testing
    test1, test2, test3 = sym.symbols(r'test_1, test_2, test_3')
    vec1 = np.matrix([1,2,3,4,5,6])
    vec2 = sym.Matrix([test1, test2, test3, test1, test2, test3])
    vec3 = np.array([1, 2, 3, 4, 5, 6])

    #inertia matrix testing
    #---------------------------#

    #print("InertiaMatrix6 tests:")

    ##not currently configured to work
    ## m1 = 4
    ## scriptI1 = 7*np.eye(3)

    m2 = sym.symbols(r'test_m')
    scriptI2 = sym.symbols(r'test_J') * sym.eye(3)
    ## print(InertiaMatrix6(m1, scriptI1))
    #display(InertiaMatrix6(m2, scriptI2))


    #---------------------------#

    mat1 = HatVector6(vec1)
    mat2 = HatVector6(vec2)
    mat3 = HatVector6(vec3)

    #print("HatVector6 tests:")
    # print(type(mat1))
    # print(mat1, end='\n\n')

    # print(type(mat2))
    # display(mat2)

    # print(type(mat3))
    # print(mat3, end='\n\n')

    #---------------------------#

    vec4 = UnhatMatrix4(mat1)
    vec5 = UnhatMatrix4(mat2)
    vec6 = UnhatMatrix4(mat3)

    # print("UnhatMatrix4 tests:")
    # print(type(vec4))
    # print(vec4, end='\n\n')

    # print(type(vec5))
    # display(vec5)

    # print(type(vec6))
    # print(vec6, end='\n\n')

    pass

def TestVb6():
    #testing
    t = sym.symbols(r't')
    x = sym.Function(r'x')(t)
    y = sym.Function(r'y')(t)
    theta1 = sym.Function(r'\theta_1')(t)
    theta2 = sym.Function(r'\theta_2')(t)

    R = sym.Matrix([
        [sym.cos(-theta2), -sym.sin(-theta2), 0],
        [sym.sin(-theta2),  sym.cos(-theta2), 0],
        [              0,                0,   1]

    ])

    G = SOnAndRnToSEn(R, [x,y,0])
    V = CalculateVb6(G,t)
    print("\nV:")
    display(V)

def TestSEn():

    #test cases

    #SO(2) and R2 - numpy
    mat1 = np.matrix([[1,2],[3,4]])
    p1 = [5,6]
    out = SOnAndRnToSEn(mat1, p1)
    assert np.array_equal(out,  np.matrix([[1,2,5],[3,4,6],[0,0,1]]) ), f"{out}"

    #SO(2) and R2 - sympy
    mat2 = sym.Matrix([[5,6],[7,8]])
    p2 = [9,0]
    out = SOnAndRnToSEn(mat2, p2)
    assert out - sym.Matrix([[5,6,9],[7,8,0],[0,0,1]]) == sym.zeros(3,3), f"{out}"

    #SO(3) and R3 - numpy 
    mat3 = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    p3 = [1.1,2.2,3.3]
    out = SOnAndRnToSEn(mat3, p3)
    assert np.array_equal(out,  np.matrix([[1,2,3,1.1],[4,5,6,2.2],[7,8,9,3.3],[0,0,0,1]]) ), f"{out}"

    #SO(3) and R3 - sympy 
    mat4 = sym.Matrix([[1,2,3],[4,5,6],[7,8,9]])
    p4 = [4.4,5.5,6.6]
    out = SOnAndRnToSEn(mat4, p4)
    diff = out - sym.Matrix([[1,2,3,4.4],[4,5,6,5.5],[7,8,9,6.6],[0,0,0,1]])
    assert diff == sym.zeros(4,4), f"{out}\n\n{diff}"

    #dimensional mismatch - check that it throws an error
    #SOnAndRnToSEn(mat2, p4)

    #type mismatch - check that it throws an error
    #SOnAndRnToSEn(mat2, sym.Matrix(p1))
    #SOnAndRnToSEn(mat1, np.matrix(p2))

    #SE(3)
    SE3mat = SOnAndRnToSEn(np.identity(3), [1,2,3])
    [SO3, R3] = SEnToSOnAndRn(SE3mat)
    assert np.array_equal(SO3, np.identity(3)) and np.array_equal(R3, [1,2,3]), f"{SO3}\n{R3}"

    #SE(2)
    SE3mat = SOnAndRnToSEn(np.identity(2), [4,5])
    [SO2, R2] = SEnToSOnAndRn(SE3mat)
    assert np.array_equal(SO2, np.identity(2)) and np.array_equal(R2, [4,5]), f"{SO2}\n{R2}"

    print("All assertions passed")


if __name__ == '__main__':
    #dill_test(lagrangian)
    TestHat3()
    TestMatrix4()
    TestVb6()
    TestSEn()
