from py_ecc.bn128 import pairing, FQ12, neg, final_exponentiate, Z1, add, multiply
from functools import reduce
from typing import Dict

class Verifier:

    def __init__(self, setup: Dict):
        
        # unpack secret parameters from setup
        self.alpha_G1 = setup["alpha_G1"]
        self.beta_G2 = setup["beta_G2"]
        self.delta_G2 = setup["delta_G2"]
        self.gamma_G2 = setup["gamma_G2"]
        self.pub_powers_of_tau_G1 = setup["pub_powers_of_tau_G1"]

    def verifyProof(self, A, B, C, public_input):

        # NOTE: G12 points
        a = pairing(B, neg(A))
        b = pairing(self.beta_G2, self.alpha_G1)
        c = pairing(self.gamma_G2, self.elliptic_dot(self.pub_powers_of_tau_G1, public_input))
        d = pairing(self.delta_G2, C)

        result = final_exponentiate(a * b * c * d) == FQ12.one()

        print("Proof is Valid: ", result)

        return result
    
    def elliptic_dot(self, ec_pts, coeffs): 
        # elliptic curve dot product for tau powers with poly coefficients 
        return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)
    