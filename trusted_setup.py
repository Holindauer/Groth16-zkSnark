import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply, field_modulus, curve_order

class TrustedSetup:

    def secret_tau(self):
        return random.randint(1, field_modulus)
    
    def G1_powers(self, tau, n):
        return [multiply(G1, tau**i) for i in range(n)]
    
    def G2_powers(self, tau, n):
        return [multiply(G2, tau**i) for i in range(n)]



if __name__ == "__main__":
    trusted_setup = TrustedSetup()

    tau = trusted_setup.secret_tau()

    G1_powers = trusted_setup.G1_powers(tau, 4)
    G2_powers = trusted_setup.G2_powers(tau, 4)

    print("G1_powers: \n", G1_powers, "\n")

