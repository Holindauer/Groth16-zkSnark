import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply, curve_order #, field_modulus
import galois
from functools import reduce

class TrustedSetup: 

    def setup( degree):

        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        # GF = galois.GF(curve_order) 
        # field_modulus = field_modulus
        GF = galois.GF(79)
        field_modulus = 79
    
        # create t = (x - 1)(x - 2) * ... * (x - n)
        x_min_i = [galois.Poly([1, field_modulus - i], field = GF) for i in range(1, degree)]
        t = reduce(lambda x, y: x * y, x_min_i)

        # set random tau
        tau = secret_tau(field_modulus)

        # generate G1 and G2 powers for polynomial evaluation
        G1_powers, G2_powers = tau_powers(tau, degree)

        # eval t at G1 powers
        t_G1 = t_eval_G1_powers(t, G1_powers)

            
        return G1_powers, G2_powers, t_G1


def secret_tau(field_modulus):
    return random.randint(1, field_modulus)

def tau_powers( tau, poly_degree):
    G1_powers = [multiply(G1, tau**i) for i in range(poly_degree)]
    G2_powers = [multiply(G2, tau**i) for i in range(poly_degree)]
    return G1_powers, G2_powers

def encrypted_dot(ec_pts, coeffs): 
    return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)

def t_eval_G1_powers(t, G1_powers):
    return encrypted_dot(G1_powers, t.coeffs)
        





