import numpy as np

# polynomial constraint: out = x^3 - 4x^2 + 6x + y^2 - 40

# R1CS matrices --- Ls * Rs = Os ,[1, out, x, y, v1, v2, v3]
L = np.array([[0,0,1,0,0,0,0],
              [0,0,0,0,1,0,0],
              [0,0,4,0,0,0,0],
              [0,0,0,1,0,0,0]])

R = np.array([[0,0,1,0,0,0,0],
              [0,0,1,0,0,0,0],
              [0,0,1,0,0,0,0],
              [0,0,0,1,0,0,0]])

O = np.array([[0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1],
              [40,0,-6,0,0,-1,1]])


# inputs that solve the polynomial constraint
x, y = 4, 4

# Original Constraint:
# out = x^3 - 4x^2 + 6x + y^2 - 40

# Transformed Constraint:
# v1 = x * x
# v2 = v1 * x
# v3 = (4*x) * x
# -v2 + v3 - 6x + 40 + out = y*y 

# intermediate variables
v1 = x * x
v2 = v1 * x
v3 = (4*x) * x
out = v2 - v3 + 6*x + y*y

# witness vector
w = np.array([1, out, x, y, v1, v2, v3])

# # ensure that the constraint is satisfied when represented as an r1cs
result = O.dot(w) == np.multiply(L.dot(w),R.dot(w))
assert result.all(), "result contains an inequality"

