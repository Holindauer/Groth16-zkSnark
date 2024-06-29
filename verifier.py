from py_ecc.bn128 import pairing, FQ12, neg, final_exponentiate

class Verifier:

    def verifyProof(self, A, B, C, public_input):

        # NOTE: G12 points
        a = pairing(B, neg(A))
        b = pairing(self.beta_G2, self.alpha_G1)
        c = pairing(self.gamma_G2, self.elliptic_dot(self.pub_powers_of_tau_G1, public_input))
        d = pairing(self.delta_G2, C)

        result = final_exponentiate(a * b * c * d) == FQ12.one()

        print("Proof is Valid: ", result)