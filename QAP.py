import numpy as np
from scipy.interpolate import lagrange
from py_ecc.bn128 import G1, G2, multiply, add, curve_order, field_modulus, eq, Z1, pairing
import galois
from functools import reduce


# R1CS matrices --- Ls * Rs = Os ,[1, out, x, y, v1inter, v1, v2]
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

print("L: \n", L, "\n")
print("R: \n", R, "\n")
print("O: \n", O, "\n")

# galois field w/ modulus 79
# field_modulus = 79
GF = galois.GF(field_modulus)


# inputs that solve the polynomial constraint
x = GF(3)
y = GF(2)

# original polynomial
out = x*x*x + 4*x*x + y*y

# intermediate variables
v1inter = x*x
v1 = v1inter*x
v2 = 4*x*x

# witness vector
w = np.array([1, out, x, y, v1inter, v1, v2])


print("witness: \n", w, "\n")

def encode_array(arr: np.array):
    # convert np arr to galois field arr, handling negatives
    return GF(np.array(arr, dtype=int) % field_modulus)
 
L_galois = encode_array(L)
R_galois = encode_array(R)
O_galois = encode_array(O)
w_galois = encode_array(w)


print("L_galois: \n", L_galois, "\n")
print("R_galois: \n", R_galois, "\n")
print("O_galois: \n", O_galois, "\n")


def poly_interpolate_column(col):
    xs = GF(np.array([1,2,3,4])) # interpolate over 1,2,3,4
    return galois.lagrange_poly(xs, col)

# axis 0 is the columns. apply_along_axis is the same as doing a 
# for loop over the columns and collecting the results in an array
U_polys = np.apply_along_axis(poly_interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(poly_interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(poly_interpolate_column, 0, O_galois)


print("U_polys: \n", U_polys, "\n")
print("V_polys: \n", V_polys, "\n")
print("W_polys: \n", W_polys, "\n")

print(U_polys[:2])
print(V_polys[:2])
print(W_polys[:1])


# U_dot_s * V_dot_s = W_dot_s + h(x)t(x)

# for dots within QAP formula
def dot_polynomials_with_witness(polys, witness):
    mul_ = lambda x, y: x * y
    sum_ = lambda x, y: x + y
    return reduce(sum_, map(mul_, polys, witness))


U_dot_s = dot_polynomials_with_witness(U_polys, w_galois)
V_dot_s = dot_polynomials_with_witness(V_polys, w_galois)
W_dot_s = dot_polynomials_with_witness(W_polys, w_galois)


# t = (x - 1)(x - 2)(x - 3)(x - 4)
x_min_1 = galois.Poly([1, field_modulus-1], field = GF)
x_min_2 = galois.Poly([1, field_modulus-2], field = GF)
x_min_3 = galois.Poly([1, field_modulus-3], field = GF)
x_min_4 = galois.Poly([1, field_modulus-4], field = GF)

t = x_min_1 * x_min_2 * x_min_3 * x_min_4

# h(x) = (W_dot_s - U_dot_s * V_dot_s) / t
h = (U_dot_s * V_dot_s - W_dot_s) // t


print("t: \n", t, "\n")
print("h: \n", h, "\n")
print("h(x) * t(x): \n", h * t, "\n")
print("U_dot_s * V_dot_s - W_dot_s: \n", U_dot_s * V_dot_s - W_dot_s, "\n")

# this is true iff the constraint is satisfied
assert U_dot_s * V_dot_s == W_dot_s + (h * t), "division has a remainder"