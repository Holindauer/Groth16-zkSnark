import numpy as np
from py_ecc.bn128 import G1, G2, multiply, add, Z1, pairing
import galois
from functools import reduce
from trusted_setup import TrustedSetup
from typing import List
 
class Prover:
    def __init__(self, L, R, O, setup: List):

        # r1cs matrices
        self.L, self.R, self.O = L, R, O    

        # unpack setup
        self.GF = setup[0]
        self.curve_order = setup[1]
        self.G1_tau_powers = setup[2]
        self.G2_tau_powers = setup[3]
        self.t_G1 = setup[4]

        # flags
        self.ensure_valid_proof = True
        self.verbose = True

    def genProof(self, x, y):

        # gen wintess
        witness = self.genWitness(x, y)  
        if self.ensure_valid_proof:
            self.verifyR1CS(self.L, self.R, self.O, witness)

        # convert R1CS to QAP
        U_dot_s, V_dot_s, W_dot_s, h, t = self.R1CS_to_QAP(self.L, self.R, self.O, witness)
        if self.ensure_valid_proof:
            self.verifyQAP(U_dot_s, V_dot_s, W_dot_s, h, t)

        # evaluate polynomials at encrypted powers of tau
        U_eval = self.eval_polys_at_encrypted_tau(self.G1_tau_powers, U_dot_s)
        V_eval = self.eval_polys_at_encrypted_tau(self.G2_tau_powers, V_dot_s) # NOTE: G2
        W_eval = self.eval_polys_at_encrypted_tau(self.G1_tau_powers, W_dot_s)

        # evaluate h at t(G1_powers)
        ht_eval = self.eval_ht(h, self.t_G1)

        # compute A, B, C
        A, B, C = U_eval, V_eval, add(W_eval, ht_eval)
        if self.ensure_valid_proof:
            self.verify_pairing(A, B, C)



        
    def genWitness(self, x, y):
        if self.verbose: print("generating witness...")
        # inputs that solve the polynomial constraint
        x, y = self.GF(x), self.GF(y)
        v1 = x*x
        v2 = v1*x
        v3 = 4*x*x
        out = v1 + v2 + y*y
        # witness vector as a galois field array
        return self.GF(np.array([1, out, x, y, v1, v2, v3]))
    
    def verifyR1CS(self, L, R, O, w):
        if self.verbose: print("ensuring witness satisfies R1CS constraint...")
        result = O.dot(w) == np.multiply(L.dot(w), R.dot(w))
        assert result.all(), "result contains an inequality"
           
    def R1CS_to_QAP(self, L, R, O, w):
        if self.verbose: print("converting R1CS to QAP...")

        # interpolate matrices
        U_polys, V_polys, W_polys, s_galois = self.poly_interpolate_matrices(L, R, O, w)
        
        # U_dot_s, V_dot_s, W_dot_s 
        U_dot_s, V_dot_s, W_dot_s = [self.dot(poly , s_galois) for poly in [U_polys, V_polys, W_polys]]

        # compute h and t
        t = self.compute_t(degree=4)
        h = self.compute_h(U_dot_s, V_dot_s, W_dot_s, t)

        # true iff the constraint is satisfied
        return U_dot_s, V_dot_s, W_dot_s, h, t
    
    def verifyQAP(self, U_dot_s, V_dot_s, W_dot_s, h, t):
        if self.verbose: print("verifying QAP...")
        assert U_dot_s * V_dot_s == W_dot_s + h * t, "division has a remainder"

    def encode_array(self, arr: np.array):
        # convert np arr to galois field arr, handling negatives
        return self.GF(np.array(arr, dtype=int) % self.curve_order)
    
    def poly_interpolate_matrices(self, L, R, O, w):
        if self.verbose: print("interpolating R1CS matrices...")

        # # convert np arr to galois field arr, handling negatives
        encode_array = lambda arr: self.GF(np.array(arr, dtype=int) % self.curve_order)
        L_galois, R_galois, O_galois, w_galois = [self.encode_array(arr) for arr in [L, R, O, w]]

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
        if self.verbose: print("computing t...")
        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, self.curve_order - i], field = self.GF) for i in range(1, 5)]
        return reduce(lambda x, y: x * y, x_min_i) # NOTE: poly expansion
    
    def compute_h(self, U_dot_s, V_dot_s, W_dot_s, t):  
        if self.verbose: print("computing h...")
        # h(x) = (W_dot_s - U_dot_s * V_dot_s) 
        return (U_dot_s * V_dot_s - W_dot_s) // t  # NOTE: poly expansion
    
    def eval_polys_at_encrypted_tau(self, powers_of_tau, poly):
        if self.verbose: print("evaluating polynomials at encrypted tau...")
        # elliptic curve dot product for tau powers with poly coefficients 
        return self.elliptic_dot(powers_of_tau, poly.coeffs[::-1])
    
    def eval_ht(self, h, t_G1):
        if self.verbose: print("evaluating h at t(G1 powers)")
        # evaluate h at G1 powers
        return self.elliptic_dot(t_G1[:3], h.coeffs[::-1]) # NOTE: only 3 coefficients on h
    
    def verify_pairing(self, A, B, C):
        assert(pairing(B, A) == pairing(G2, C)), "pairing check failed"


if __name__ == "__main__":

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
    
    # trusted setup
    setup = TrustedSetup.setup(degree=4)

    
    prover = Prover(L, R, O, setup)

    prover.genProof(3, 2)