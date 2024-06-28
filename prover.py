import numpy as np
from py_ecc.bn128 import G1, G2, multiply, add, Z1, pairing
import galois
from functools import reduce
from trusted_setup import TrustedSetup
from QAP import QAP
from typing import Dict
import random
 
class Prover:
    def __init__(self, U_poly, V_poly, W_poly, setup: Dict):

        # rQAP
        self.U, self.V, self.W = U_poly, V_poly, W_poly
        
        # unpack galois field w/ matching modulus to bn128 from setup
        self.GF = setup["GF"]
        self.curve_order = setup["curve_order"]

        # unpack secret parameters from setup
        self.alpha_G1 = setup["alpha_G1"]
        self.beta_G1 = setup["beta_G1"]
        self.beta_G2 = setup["beta_G2"]
        self.delta_G1 = setup["delta_G1"]
        self.delta_G2 = setup["delta_G2"]
        self.upsilon_G2 = setup["upsilon_G2"]
        self.A_powers_of_tau_G1 = setup["A_powers_of_tau_G1"]
        self.B_powers_of_tau_G2 = setup["B_powers_of_tau_G2"]   
        self.public_powers_of_tau_G1 = setup["public_powers_of_tau_G1"]
        self.private_powers_of_tau_G1 = setup["private_powers_of_tau_G1"]
        self.ht_powers_of_tau_G1 = setup["ht_powers_of_tau_G1"]

        # pre-evaluate t at powers of tau
        self.t_G1 = setup["t_G1"]

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

        # generate r and s 
        r, s = [self.random_field_element() for _ in range(2)]

        # compute h(tau)t(tau)
        ht_G1 = self.eval_ht(h)

        # # compute A, B, C
        A_G1 = self.compute_A_G1(U_dot_s, r)
        B_G2 = self.compute_B_G2(V_dot_s, s)
        B_G1 = self.compute_B_G1(V_dot_s, s)
        
        C_G1 = self.compute_C_G1(witness, A_G1, B_G1, ht_G1, s, r)


        A, B, C = A_G1, B_G2, C_G1

        if self.verbose:
            print("Proof generated successfully!")
            print("A: ", A)
            print("B: ", B)
            print("C: ", C)

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
    
    def random_field_element(self):
        return self.GF(random.randint(1, self.curve_order))
    
    def elliptic_dot(self, ec_pts, coeffs): 
        # elliptic curve dot product for tau powers with poly coefficients 
        return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)
    
    def eval_ht(self, h):
        # evaluate h at G1 powers
        return self.elliptic_dot(self.ht_powers_of_tau_G1[:3], h.coeffs[::-1]) # NOTE: only 3 coefficients on h

    def compute_A_G1(self, U_dot_s, r):
        # A = alpha_G1 + U_dot_s(tau) + (r * delta_G1)
        secret_shift = self.alpha_G1
        U_eval = self.elliptic_dot(self.A_powers_of_tau_G1, U_dot_s.coeffs[::-1])
        r_mul_d = multiply(self.delta_G1, int(r))
        return (add(secret_shift, add(U_eval, r_mul_d)))
    
    def compute_B_G2(self, V_dot_s, s):
        # B = beta_G2 + V_eval + (s * delta_G2)
        secret_shift = self.beta_G2
        V_eval = self.elliptic_dot(self.B_powers_of_tau_G2, V_dot_s.coeffs[::-1])
        s_mul_d = multiply(self.delta_G2, int(s))
        return (add(secret_shift, add(V_eval, s_mul_d)))
    
    def compute_B_G1(self, V_dot_s, s):
        # B = beta_G1 + V_eval + (s * delta_G1)
        secret_shift = self.beta_G1
        V_eval = self.elliptic_dot(self.A_powers_of_tau_G1, V_dot_s.coeffs[::-1])
        s_mul_d = multiply(self.delta_G1, int(s))
        return (add(secret_shift, add(V_eval, s_mul_d)))

    def compute_C_G1(self, witness, A_G1, B_G1, ht, s, r):
        # NOTE: C computed over private inputs

        term_1 = self.elliptic_dot(self.private_powers_of_tau_G1, witness)
        sA_G1 = multiply(A_G1, int(s))
        rB_G1 = multiply(B_G1, int(r))
        rsDelta_G1 = multiply(self.delta_G1, int((-r)*s)) # ! this is intended to be subtraction 

        C = add(add(add(add(term_1, ht), sA_G1), rB_G1), rsDelta_G1)

        return C

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