import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply#, curve_order
from functools import reduce
import galois
from QAP import QAP

class TrustedSetup: 

    def __init__(self, U_poly, V_poly, W_poly, t, GF, curve_order): # ! currently passing GF in to reduce dev time, but it should be generated here

        # unpack QAP
        self.U, self.V, self.W, self.t = U_poly, V_poly, W_poly, t

        # galois field w/ matching modulus to bn128
        self.GF = GF
        self.curve_order = curve_order
        # print("initializing a large field, will take a moment...")
        # GF = galois.GF(curve_order)
        # self.curve_order = curve_order 


    def setup(self, degree): 

        # generate secret parameters
        tau, alpha, beta, delta, upsilon = [self.random_field_element() for _ in range(5)]

        # generate secret shifts for A and B
        alpha_G1 = self.A_secret_shift_G1(alpha) # G1 pt
        beta_G2 = self.B_secret_shift_G2(beta)   # G2 pt

        # powers of tau for A and B
        A_powers_of_tau_G1 = self.compute_A_powers_of_tau(tau, degree)
        B_powers_of_tau_G2 = self.compute_B_powers_of_tau(tau, degree)

        # powers of tau for public and private inputs in C
        public_powers_of_tau_G1 = self.compute_C_public_powers_of_tau(tau, alpha, beta, upsilon, degree)
        private_powers_of_tau_G1 = self.compute_C_private_powers_of_tau(tau, alpha, beta, delta, degree)

        # powers of tau for h(tau)t(tau) in C
        ht_powers_of_tau_G1 = self.compute_ht_powers_of_tau(self.t, tau, delta, degree)

        # secret delta and upsilon
        delta_G2 = self.secret_delta(delta)
        upsilon_G2 = self.secret_upsilon(upsilon)


        setup = {
            "GF": self.GF,
            "curve_order": self.curve_order,
            "alpha_G1": alpha_G1,
            "beta_G2": beta_G2,
            "A_powers_of_tau_G1": A_powers_of_tau_G1,
            "B_powers_of_tau_G2": B_powers_of_tau_G2,
            "public_powers_of_tau_G1": public_powers_of_tau_G1,
            "private_powers_of_tau_G1": private_powers_of_tau_G1,
            "ht_powers_of_tau_G1": ht_powers_of_tau_G1,
            "delta_G2": delta_G2,
            "upsilon_G2": upsilon_G2
        }

        return setup


    def random_field_element(self):
        return self.GF(random.randint(1, self.curve_order))

    def A_secret_shift_G1(self, alpha):
        return multiply(G1, int(alpha)) # secret G1 shift for A

    def B_secret_shift_G2(self, beta): 
        return multiply(G2, int(beta)) # secret G2 shift for B

    def compute_A_powers_of_tau(self, tau, poly_degree):
        return [multiply(G1, int(tau**i)) for i in range(poly_degree)] # NOTE: G1
    
    def compute_B_powers_of_tau(self, tau, poly_degree):
        return [multiply(G2, int(tau**i)) for i in range(poly_degree)] # NOTE: G2
    
    def compute_C_public_powers_of_tau(self, tau, alpha, beta, upsilon, poly_degree):
        constant = lambda u, v, w, x: ((beta * u(x)) + (alpha * v(x)) + w(x)) * (upsilon**-1)
        u, v, w = self.U, self.V, self.W
        return [ multiply( G1, int(constant(u[i], v[i], w[i], (tau**i)))) for i in range(poly_degree)] # NOTE: G1
    
    def compute_C_private_powers_of_tau(self, tau, alpha, beta, delta, poly_degree):
        constant = lambda u, v, w, x: ((beta * u(x)) + (alpha * v(x)) + w(x)) * (delta**-1)
        u, v, w = self.U, self.V, self.W
        return [multiply(G1, int(constant(u[i], v[i], w[i], (tau**i)))) for i in range(poly_degree)] # NOTE: G1

    def compute_ht_powers_of_tau(self, t, tau, delta, poly_degree):
        constant = lambda tau_power: tau_power * t(tau) * (delta**-1)
        return [multiply(G1, int(constant(tau**i))) for i in range(poly_degree-2)] # NOTE: G1
    
    def secret_upsilon(self, upsilon):
        return multiply(G2, int(upsilon))
    
    def secret_delta(self, delta):
        return multiply(G2, int(delta))
    

if __name__ == "__main__":

    qap = QAP()

    # convert R1CS to QAP
    U_poly, V_poly, W_poly, t = qap.R1CS_to_QAP()

    # get galois field 
    GF = qap.GF
    curve_order = qap.curve_order

    # trusted setup
    trusted_setup = TrustedSetup(U_poly, V_poly, W_poly, t, GF, curve_order)
    
    setup = trusted_setup.setup(degree=4)
    for key, val in setup.items():
        print("\n", key, val)