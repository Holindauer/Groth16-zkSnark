import numpy as np
from py_ecc.bn128 import G1, G2, multiply, add, curve_order, field_modulus, eq, Z1, pairing
import galois
from functools import reduce
from trusted_setup import TrustedSetup
 
class Prover:
    def __init__(self, L, R, O):

        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        self.GF = galois.GF(field_modulus) 
        self.field_modulus = field_modulus
        # self.GF = galois.GF(79)
        # self.field_modulus = 79

        # r1cs matrices
        self.L, self.R, self.O = L, R, O    

    def genProof(self, x, y):

        # trusted setup
        G1_tau_powers, G2_tau_powers, t_G1 = TrustedSetup.setup(degree=4)

        # gen wintess
        witness = self.genWitness(x, y)  

        # convert r1cs to QAP
        U_dot_s, V_dot_s, W_dot_s, h = self.R1CS_to_QAP(self.L, self.R, self.O, witness)
        
        # evaluate polynomials at encrypted powers of tau
        U_eval = self.eval_polys_at_encrypted_tau(G1_tau_powers, U_dot_s)
        V_eval = self.eval_polys_at_encrypted_tau(G2_tau_powers, V_dot_s) # NOTE: G2
        W_eval = self.eval_polys_at_encrypted_tau(G1_tau_powers, W_dot_s)

        # evaluate h at t(G1_powers)
        h_eval = self.eval_ht(h, t_G1)

        A_1 = U_eval
        B_2 = V_eval
        C_1 = W_eval + h_eval

        print("A_1", A_1)
        print("B_2", B_2)
        print("C_1", C_1)
    
        print(pairing(A_1, B_2) == pairing(C_1, G2))

    def genWitness(self, x, y):
        # # inputs that solve the polynomial constraint
        # x, y = self.GF(4), self.GF(4)
        # v1 = x * x
        # v2 = v1 * x
        # v3 = (4*x) * x
        # out = v2 - v3 + 6*x + y*y

        # # witness vector
        # return np.array([1, out, x, y, v1, v2, v3])

        # original polynomial
        out = x*x*x + 4*x*x + y*y

        # intermediate variables
        v1inter = x*x
        v1 = v1inter*x
        v2 = 4*x*x

        # witness vector
        return np.array([1, out, x, y, v1inter, v1, v2])
                
    def R1CS_to_QAP(self, L, R, O, w):

        # interpolate matrices
        U_polys, V_polys, W_polys, w_galois = self.poly_interpolate_matrices(L, R, O, w)
        
        # U_dot_s, V_dot_s, W_dot_s 
        U_dot_s, V_dot_s, W_dot_s = [self.dot(poly , w_galois) for poly in [U_polys, V_polys, W_polys]]

        # compute h and t
        t = self.compute_t(4)
        h = self.compute_h(U_dot_s, V_dot_s, W_dot_s, t)

        print("h", h)
        print("h coeffs", h.coeffs[::-1])

        # true iff the constraint is satisfied
        assert U_dot_s * V_dot_s == W_dot_s + h * t, "division has a remainder"
        return U_dot_s, V_dot_s, W_dot_s, h
    

    def poly_interpolate_matrices(self, L, R, O, w):

        # convert np arr to galois field arr, handling negatives
        encode_array = lambda arr: self.GF(np.array(arr, dtype=int) % self.field_modulus)
        L_galois, R_galois, O_galois, w_galois = [encode_array(arr) for arr in [L, R, O, w]]

        # r1cs column polynomial interpolation over 1,2,3,4
        interpolate = lambda col: galois.lagrange_poly(self.GF(np.array([1,2,3,4])), col)
        U_polys = np.apply_along_axis(interpolate, 0, L_galois)
        V_polys = np.apply_along_axis(interpolate, 0, R_galois)
        W_polys = np.apply_along_axis(interpolate, 0, O_galois)
        return U_polys, V_polys, W_polys, w_galois

    def dot(self, polys, witness):
        # dot product of polynomials with witness
        mul_ = lambda x, y: x * y
        sum_ = lambda x, y: x + y
        return reduce(sum_, map(mul_, polys, witness))
    
    def elliptic_dot(self, ec_pts, coeffs): 
        # elliptic curve dot product for tau powers with poly coefficients 
        return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)
    
    def compute_t(self, degree):  
        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        # x_min_i = [galois.Poly([1, self.field_modulus - i], field = self.GF) for i in range(1, 5)]
        # return reduce(lambda x, y: x * y, x_min_i) # NOTE: poly expansion

        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_1 = galois.Poly([1, self.field_modulus-1], field = self.GF)
        x_min_2 = galois.Poly([1, self.field_modulus-2], field = self.GF)
        x_min_3 = galois.Poly([1, self.field_modulus-3], field = self.GF)
        x_min_4 = galois.Poly([1, self.field_modulus-4], field = self.GF)

        return x_min_1 * x_min_2 * x_min_3 * x_min_4
    
    def compute_h(self, U_dot_s, V_dot_s, W_dot_s, t):  
        # h(x) = (W_dot_s - U_dot_s * V_dot_s) 
        return (U_dot_s * V_dot_s - W_dot_s) // t  # NOTE: poly expansion
    
    def eval_polys_at_encrypted_tau(self, powers_of_tau, poly):
        # elliptic curve dot product for tau powers with poly coefficients 
        return self.elliptic_dot(powers_of_tau, poly.coeffs[::-1])
    
        # h coeefs h coeffs [ 9120101196599698009269335727190531286956796315540759859453765789435510920243, 17632195646759416151254049072568360488116472876712135728277280526241987779136, 15808175407439476549400181927130254230725113613603983756386527368354885595088]
    
    def eval_ht(self, h, t_G1):
        # evaluate h at G1 powers
        return self.elliptic_dot(t_G1[:3], h.coeffs[::-1])





if __name__ == "__main__":

    # R1CS matrices --- Ls * Rs = Os ,[1, out, x, y, v1, v2, v3]
    # L = np.array([[0,0,1,0,0,0,0],
    #               [0,0,0,0,1,0,0],
    #               [0,0,4,0,0,0,0],
    #               [0,0,0,1,0,0,0]])

    # R = np.array([[0,0,1,0,0,0,0],
    #               [0,0,1,0,0,0,0],
    #               [0,0,1,0,0,0,0],
    #               [0,0,0,1,0,0,0]])

    # O = np.array([[0,0,0,0,1,0,0],
    #               [0,0,0,0,0,1,0],
    #               [0,0,0,0,0,0,1],
    #               [40,0,-6,0,0,-1,1]])

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

    
    prover = Prover(L, R, O)

    prover.genProof(3, 2)