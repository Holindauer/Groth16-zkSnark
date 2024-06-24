import numpy as np
from py_ecc.bn128 import G1, multiply, add, curve_order, field_modulus, eq, Z1
import galois
from functools import reduce

class Prover:
    def __init__(self, L, R, O):

        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        # self.GF = galois.GF(curve_order) 
        # self.field_modulus = field_modulus

        self.GF = galois.GF(79)
        self.field_modulus = 79

        # r1cs matrices
        self.L, self.R, self.O = L, R, O    

    def genProof(self, x, y):

        witness = self.genWitness(x, y)  
        U_dot_s, V_dot_s, W_dot_s, h, t = self.R1CS_to_QAP(self.L, self.R, self.O, witness)

        
    def genWitness(self, x, y):
        # inputs that solve the polynomial constraint
        x, y = self.GF(4), self.GF(4)
        v1 = x * x
        v2 = v1 * x
        v3 = (4*x) * x
        out = v2 - v3 + 6*x + y*y

        # witness vector
        return np.array([1, out, x, y, v1, v2, v3])
                
    def R1CS_to_QAP(self, L, R, O, w):
        # computes terms of: U_dot_s * V_dot_s = W_dot_s + h(x)t(x)

        # convert np arr to galois field arr, handling negatives
        encode_array = lambda arr: self.GF(np.array(arr, dtype=int) % self.field_modulus)
        L_galois, R_galois, O_galois, w_galois = [encode_array(arr) for arr in [L, R, O, w]]

        # r1cs column polynomial interpolation over 1,2,3,4
        interpolate = lambda col: galois.lagrange_poly(self.GF(np.array([1,2,3,4])), col)
        U_polys = np.apply_along_axis(interpolate, 0, L_galois)
        V_polys = np.apply_along_axis(interpolate, 0, R_galois)
        W_polys = np.apply_along_axis(interpolate, 0, O_galois)

        # for dots within QAP formula
        def dot(polys, witness):
            mul_ = lambda x, y: x * y
            sum_ = lambda x, y: x + y
            return reduce(sum_, map(mul_, polys, witness))
        
        # U_dot_s, V_dot_s, W_dot_s 
        U_dot_s, V_dot_s, W_dot_s = [dot(poly , w_galois) for poly in [U_polys, V_polys, W_polys]]

        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, self.field_modulus - i], field = self.GF) for i in range(1, 4)]
        t = reduce(lambda x, y: x * y, x_min_i)

        # h(x) = (W_dot_s - U_dot_s * V_dot_s) / t
        h = (U_dot_s * V_dot_s - W_dot_s) // t

        # true iff the constraint is satisfied
        assert U_dot_s * V_dot_s == W_dot_s + h * t, "division has a remainder"

        return U_dot_s, V_dot_s, W_dot_s, h, t
            


if __name__ == "__main__":

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
    
    prover = Prover(L, R, O)

    prover.genProof(4, 4)