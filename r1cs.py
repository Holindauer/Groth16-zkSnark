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
              [67,0,0,0,0,-1,-1]])


# inputs that solve the polynomial constraint
x = 3
y = 2

# original polynomial
out = x*x*x + 4*x*x + y*y

# intermediate variables
v1inter = x*x
v1 = v1inter*x
v2 = 4*x*x

# witness vector
w = np.array([1, out, x, y, v1inter, v1, v2])

print("witness vector: ", w)


# # ensure that the constraint is satisfied when represented as an r1cs
result = O.dot(w) == np.multiply(L.dot(w),R.dot(w))
assert result.all(), "result contains an inequality"
print("R1CS constraint satisfied")
