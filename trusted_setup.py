import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply
from functools import reduce
import galois
from QAP import QAP

class TrustedSetup: 
    def __init__(self, U_poly, V_poly, W_poly, t, GF, curve_order): 
        # QAP
        self.U, self.V, self.W, self.t = U_poly, V_poly, W_poly, t
        # galois field w/ matching modulus to bn128
        self.GF, self.curve_order  = GF, curve_order

    def setup(self, degree): 

        # public/private input split point for witness [pub, pub, priv, priv, ...]
        l = 1

        # generate secret parameters
        tau, alpha, beta, delta, gamma = [self.random_field_element() for _ in range(5)]

        # secret shifts for A and B
        alpha_G1 = multiply(G1, int(alpha))
        beta_G2 = multiply(G2, int(beta))

        # powers of tay for A and B
        A_powers_of_tau_G1 = self.compute_powers_of_tau_G1(tau, degree)
        B_powers_of_tau_G2 = self.compute_powers_of_tau_G2(tau, degree)
        
        # powers of tau for h(tau)t(tau) in C
        ht_powers_of_tau_G1 = self.compute_powers_of_tau_ht(self.t, tau, delta, degree)

        # powers of tau for public and private inputs in C
        out = self.compute_pub_and_priv_powers_of_tau(tau, alpha, beta, delta, gamma, l)
        pub_powers_of_tau_G1, priv_powers_of_tau_G1 = out

        # additional secret parameters
        beta_G1 = multiply(G1, int(beta))
        delta_G1 = multiply(G1, int(delta))
        gamma_G2 = multiply(G2, int(gamma))
        delta_G2 = multiply(G2, int(delta))

        return {
            "GF": self.GF,
            "curve_order": self.curve_order,
            "alpha_G1": alpha_G1,
            "beta_G1": beta_G1,
            "beta_G2": beta_G2,
            "A_powers_of_tau_G1": A_powers_of_tau_G1,
            "B_powers_of_tau_G2": B_powers_of_tau_G2,
            "priv_powers_of_tau_G1": priv_powers_of_tau_G1,
            "pub_powers_of_tau_G1": pub_powers_of_tau_G1,
            "ht_powers_of_tau_G1": ht_powers_of_tau_G1,
            "delta_G1": delta_G1,   
            "delta_G2": delta_G2,
            "gamma_G2": gamma_G2,
        }

    def random_field_element(self):
        return self.GF(random.randint(1, self.curve_order))

    def compute_powers_of_tau_G1(self, tau, poly_degree):
        return [multiply(G1, int(tau**i)) for i in range(poly_degree)] # NOTE: G1
    
    def compute_powers_of_tau_G2(self, tau, poly_degree):
        return [multiply(G2, int(tau**i)) for i in range(poly_degree)] # NOTE: G2

    def compute_powers_of_tau_ht(self, t, tau, delta, poly_degree):
        # powers of tau for h(tau)t(tau) in C
        t_at_tau = t(tau)
        delta_inv = self.GF(1) / delta
        out = [multiply(G1, int((tau**i) * t_at_tau)) for i in range(t.degree-1)] # NOTE: G1
        return [multiply(e, int(delta_inv)) for e in out]
    
    def compute_pub_and_priv_powers_of_tau(self, tau, alpha, beta, delta, gamma, l):

        # powers of tau for public and private inputs in C
        beta_times_U = [beta * self.U[i] for i in range(len(self.U))]
        alpha_times_V = [alpha * self.V[i] for i in range(len(self.V))]
        
        # powers of tau for C
        C_poly = [beta_times_U[i] + alpha_times_V[i] + self.W[i] for i in range(len(self.W))]
        C_poly_of_tau = [C_poly[i](tau) for i in range(len(C_poly))]
        C_powers_of_tau_G1 = [multiply(G1, int(C_poly_of_tau[i])) for i in range(len(C_poly_of_tau))]

        # powers of tau for public inputs in C
        gamma_inv = self.GF(1) / gamma
        pub_powers_of_tau_G1 = C_powers_of_tau_G1[:l+1]
        pub_powers_of_tau_G1 = [multiply(e, int(gamma_inv)) for e in pub_powers_of_tau_G1]

        # powers of tau for private inputs in C
        delta_inv = self.GF(1) / delta
        priv_powers_of_tau_G1 = C_powers_of_tau_G1[l+1:]
        priv_powers_of_tau_G1 = [multiply(e, int(delta_inv)) for e in priv_powers_of_tau_G1]

        return pub_powers_of_tau_G1, priv_powers_of_tau_G1





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