import sympy as sym
from sympy.abc import x, y, z, t

# define a function as a SymPy expression
f = x**t
print('original function f: ')
display(f)

# compute the derivative of f wrt to x
dfdx = f.diff(x)
print('derivative of f wrt x: ')
display(dfdx)

# compute the derivative of f wrt to t
dfdt = f.diff(t)
print('derivative of f wrt t: ')
display(dfdt)
