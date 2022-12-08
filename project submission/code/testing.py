import numpy as np
import sympy as sym
import dill
import time
from tqdm import tqdm

from geometry import *
from helpers import *

from IPython.display import display

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
