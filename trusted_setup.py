import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply, curve_order, field_modulus
import galois
from functools import reduce

class TrustedSetup: 

    def setup( degree):

        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        GF = galois.GF(field_modulus) 
        # field_modulus = field_modulus
        # GF = galois.GF(79)
        # field_modulus = 79
    
        # create t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, field_modulus - i], field = GF) for i in range(1, 5)]
        t = reduce(lambda x, y: x * y, x_min_i)

        # set random tau
        tau = secret_tau(GF, field_modulus)

        # generate G1 and G2 powers for polynomial evaluation
        G1_powers, G2_powers = tau_powers(tau, degree)

        # eval t at G1 powers
        t_G1 = t_eval_G1_powers(t, tau, poly_degree=3)

            
        return G1_powers, G2_powers, t_G1


def secret_tau(GF, field_mod):
    return GF(random.randint(1, field_mod))

def tau_powers( tau, poly_degree):
    G1_powers = [multiply(G1, int(tau**i)) for i in range(poly_degree)]
    G2_powers = [multiply(G2, int(tau**i)) for i in range(poly_degree)]
    return G1_powers, G2_powers

def encrypted_dot(ec_pts, coeffs): 
    return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)

def t_eval_G1_powers(t, tau, poly_degree):
    # encrypt pre evaluated t at powers of tau
    return [multiply(G1, int(t(tau) * (tau**i))) for i in range(poly_degree)]
        





