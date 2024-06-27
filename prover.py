import numpy as np
from py_ecc.bn128 import G1, G2, multiply, add, Z1, pairing
import galois
from functools import reduce
from trusted_setup import TrustedSetup
from QAP import QAP
from typing import Dict
 
class Prover:
    def __init__(self, U_poly, V_poly, W_poly, setup: Dict):

        # rQAP
        self.U, self.V, self.W = U_poly, V_poly, W_poly
        
        # unpack galois field w/ matching modulus to bn128 from setup
        self.GF = setup["GF"]
        self.curve_order = setup["curve_order"]

        # unpack secret parameters from setup
        self.alpha_G1 = setup["alpha_G1"]
        self.beta_G2 = setup["beta_G2"]
        self.delta_G2 = setup["delta_G2"]
        self.upsilon_G2 = setup["upsilon_G2"]
        self.A_powers_of_tau_G1 = setup["A_powers_of_tau_G1"]
        self.B_powers_of_tau_G2 = setup["B_powers_of_tau_G2"]   
        self.public_powers_of_tau_G1 = setup["public_powers_of_tau_G1"]
        self.private_powers_of_tau_G1 = setup["private_powers_of_tau_G1"]
        self.ht_powers_of_tau_G1 = setup["ht_powers_of_tau_G1"]

        # flags
        self.ensure_valid_proof = True
        self.verbose = True

    def genProof(self, x, y):

        witness = self.genWitness(x, y)

        # dot QAP matrices coefficients w/ solution vector 
        U_dot_s, V_dot_s, W_dot_s = [self.dot(poly , witness) for poly in [self.U, self.V, self.W]]

        # compute t and h
        t = self.compute_t(degree=4)
        h = self.compute_h(U_dot_s, V_dot_s, W_dot_s, t)
        
    
        if self.ensure_valid_proof:
            self.verifyQAP(U_dot_s, V_dot_s, W_dot_s, h, t)

        # ! NOTE below needs to be modified to support the new trusted setup
        # # evaluate polynomials at encrypted powers of tau
        # U_eval = self.eval_polys_at_encrypted_tau(self.G1_tau_powers, U_dot_s)
        # V_eval = self.eval_polys_at_encrypted_tau(self.G2_tau_powers, V_dot_s) # NOTE: G2
        # W_eval = self.eval_polys_at_encrypted_tau(self.G1_tau_powers, W_dot_s)

        # # evaluate h at t(G1_powers)
        # ht_eval = self.eval_ht(h, self.t_G1)

        # # compute A, B, C
        # A, B, C = U_eval, V_eval, add(W_eval, ht_eval)
        # if self.ensure_valid_proof:
        #     self.verify_pairing(A, B, C)

    def genWitness(self, x, y):
        # inputs that solve the polynomial constraint
        x, y = self.GF(x), self.GF(y)
        v1 = x*x
        v2 = v1*x
        v3 = 4*x*x
        out = v1 + v2 + y*y
        # witness vector as a galois field array
        return self.GF(np.array([1, out, x, y, v1, v2, v3]))
    
    def verifyR1CS(self, w):
        result = self.O.dot(w) == np.multiply(self.L.dot(w), self.R.dot(w))
        assert result.all(), "result contains an inequality"

    def dot(self, polys, witness):
        # dot product of polynomials with witness
        mul_ = lambda x, y: x * y
        sum_ = lambda x, y: x + y
        return reduce(sum_, map(mul_, polys, witness))
        
    def compute_t(self, degree):  
        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, self.curve_order - i], field = self.GF) for i in range(1, 5)]
        return reduce(lambda x, y: x * y, x_min_i) # NOTE: poly expansion
    
    def compute_h(self, U_dot_s, V_dot_s, W_dot_s, t):  
        # h(x) = (W_dot_s - U_dot_s * V_dot_s) / t
        return (U_dot_s * V_dot_s - W_dot_s) // t  # NOTE: poly expansion
    
    def verifyQAP(self, U_dot_s, V_dot_s, W_dot_s, h, t):
        assert U_dot_s * V_dot_s == W_dot_s + h * t, "division has a remainder"
    
    def elliptic_dot(self, ec_pts, coeffs): 
        # elliptic curve dot product for tau powers with poly coefficients 
        return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)
    
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

    qap = QAP()

    # convert R1CS to QAP
    U_poly, V_poly, W_poly, t = qap.R1CS_to_QAP()

    # get galois field 
    GF = qap.GF
    curve_order = qap.curve_order

    # trusted setup
    ts = TrustedSetup(U_poly, V_poly, W_poly, t, GF, curve_order)
    setup = ts.setup(degree=4)
    

    prover = Prover(U_poly, V_poly, W_poly, setup)

    prover.genProof(3, 2)