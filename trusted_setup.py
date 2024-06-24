import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply, field_modulus, curve_order

class TrustedSetup: 

    def setup( degree):
        
        @staticmethod
        def secret_tau():
            return random.randint(1, field_modulus)
        
        @staticmethod
        def tau_powers( tau, poly_degree):
            G1_powers = [multiply(G1, tau**i) for i in range(poly_degree)]
            G2_powers = [multiply(G2, tau**i) for i in range(poly_degree)]
            return G1_powers, G2_powers
        
        tau = secret_tau()
        G1_powers, G2_powers = tau_powers(tau, degree)
            
        return G1_powers, G2_powers


if __name__ == "__main__":
    trusted_setup = TrustedSetup()

    tau = trusted_setup.secret_tau()

    G1_powers = trusted_setup.G1_powers(tau, 4)
    G2_powers = trusted_setup.G2_powers(tau, 4)

    print("G1_powers: \n", G1_powers, "\n")

