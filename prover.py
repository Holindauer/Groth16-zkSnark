import numpy as np
from py_ecc.bn128 import G1, G2, multiply, add, Z1, pairing, neg, final_exponentiate, FQ12
import galois
from functools import reduce
from trusted_setup import TrustedSetup
from QAP import QAP
from typing import Dict
import random
 
class Prover:
    def __init__(self, U_poly, V_poly, W_poly, setup: Dict):

        # QAP
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
        self.gamma_G2 = setup["gamma_G2"]
        self.A_powers_of_tau_G1 = setup["A_powers_of_tau_G1"]
        self.B_powers_of_tau_G2 = setup["B_powers_of_tau_G2"]   
        self.pub_powers_of_tau_G1 = setup["pub_powers_of_tau_G1"]
        self.priv_powers_of_tau_G1 = setup["priv_powers_of_tau_G1"]
        self.ht_powers_of_tau_G1 = setup["ht_powers_of_tau_G1"]

        # flags
        self.ensure_valid_proof = True
        self.verbose = True

    def genProof(self, x, y):
        
        # generate witness vector
        witness = self.genWitness(x, y)

        print("witness vector: ", witness)

        # split witness into public/private input at idx l [pub, pub, priv, priv, ...]
        l  = 1
        private_input = witness[l+1:]
        public_input = witness[:l+1]

        print("public input: ", public_input)
        print("private input: ", private_input)

        
        # dot QAP matrices coefficients w/ solution vector 
        U_dot_s, V_dot_s, W_dot_s = [self.dot(poly , witness) for poly in [self.U, self.V, self.W]]

        # compute t and h
        t = self.compute_t(degree=4)
        h = self.compute_h(U_dot_s, V_dot_s, W_dot_s, t)

        # verify R1CS to QAP conversion success
        if self.ensure_valid_proof:
            self.verifyQAP(U_dot_s, V_dot_s, W_dot_s, h, t)

        # compute initial versions of A, B, C
        A_old_G1 = self.elliptic_dot(self.A_powers_of_tau_G1, U_dot_s.coeffs[::-1])
        B_old_G2 = self.elliptic_dot(self.B_powers_of_tau_G2, V_dot_s.coeffs[::-1])
        B_old_G1 = self.elliptic_dot(self.A_powers_of_tau_G1, V_dot_s.coeffs[::-1])
        ht_of_tau_G1 = self.elliptic_dot(self.ht_powers_of_tau_G1[:3], h.coeffs[::-1])
        C_old_G1 = self.elliptic_dot(self.priv_powers_of_tau_G1, private_input)

        # generate random r and s
        r, s = [self.random_field_element() for _ in range(2)]

        # compute A
        A_G1 = add(add(A_old_G1, self.alpha_G1), multiply(self.delta_G1, int(r)))

        # compute B
        B_G2 = add(add(B_old_G2, self.beta_G2), multiply(self.delta_G2, int(s)))
        B_G1 = add(add(B_old_G1, self.beta_G1), multiply(self.delta_G1, int(s)))

        # compute C
        C_term_1 = add(C_old_G1, ht_of_tau_G1)
        C_term_2 = multiply(A_G1, int(s))
        C_term_3 = multiply(B_G1, int(r))
        C_term_4 = neg(multiply(self.delta_G1, int(r*s)))
        C_G1 = add(add(add(C_term_1, C_term_2), C_term_3), C_term_4)

        proof = [A_G1, B_G2, C_G1]
        self.printProof(proof, public_input)

        if self.ensure_valid_proof:
            
            a = pairing(B_G2, neg(A_G1))
            b = pairing(self.beta_G2, self.alpha_G1)
            c = pairing(self.gamma_G2, self.elliptic_dot(self.pub_powers_of_tau_G1, public_input))
            d = pairing(self.delta_G2, C_G1)

            result = final_exponentiate(a * b * c * d) == FQ12.one()
            print("Proof verification: ", result)

        return proof, public_input

    def genWitness(self, x, y):
        # inputs that solve the polynomial constraint
        x, y = self.GF(x), self.GF(y)
        v1 = x*x
        v2 = v1*x
        v3 = 4*x*x
        out = v1 + v2 + y*y
        # witness vector as a galois field array
        return self.GF(np.array([1, out, x, y, v1, v2, v3]))
    
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
        # verify R1CS to QAP conversion success
        assert U_dot_s * V_dot_s == W_dot_s + h * t, "division has a remainder"

    def elliptic_dot(self, ec_pts, coeffs): 
        # elliptic curve dot product for tau powers with poly coefficients 
        return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)
    
    def random_field_element(self):
        return self.GF(random.randint(1, self.curve_order))
    
    def printProof(self, proof, public_input):
        if self.verbose:
            print("Proof generated successfully")
            print("\nA: ", proof[0])
            print("\nB: ", proof[1])
            print("\nC: ", proof[2])
            print("\nPublic Claim", public_input)


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