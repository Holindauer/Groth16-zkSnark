import random
import numpy as np
from py_ecc.bn128 import G1, G2, Z1, add, multiply, curve_order
from functools import reduce
import galois

class TrustedSetup: 

    def setup( degree):

        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        GF = galois.GF(curve_order) 
        # GF = galois.GF(79)
        # curve_order = 79
    
        # create t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, curve_order - i], field = GF) for i in range(1, 5)]
        t = reduce(lambda x, y: x * y, x_min_i)

        # set random tau
        tau = secret_tau(GF, curve_order)

        # generate secret shifts for A and B
        alpha = secret_alpha(GF, curve_order)  # G1 pt
        beta = secret_beta(GF, curve_order)    # G2 pt

        # generate G1 and G2 powers for polynomial evaluation
        G1_powers, G2_powers = tau_powers(tau, degree)

        # eval t at G1 powers
        t_G1 = t_eval_G1_powers(t, tau, poly_degree=3)

            
        return GF, curve_order, G1_powers, G2_powers, t_G1, alpha, beta


def secret_tau(GF, order):
    # computes secret tau term
    return GF(random.randint(1, order))

def secret_alpha(GF, order):
    # computes secret shift for A
    random_field_element = GF(random.randint(1, order))
    return multiply(G1, int(random_field_element))

def secret_beta(GF, order): 
    # computes secret shift for B
    random_field_element = GF(random.randint(1, order))
    return multiply(G2, int(random_field_element))

def tau_powers( tau, poly_degree):
    # generates G1 and G2 powers of tau for polynomial evaluation
    G1_powers = [multiply(G1, int(tau**i)) for i in range(poly_degree)]
    G2_powers = [multiply(G2, int(tau**i)) for i in range(poly_degree)]
    return G1_powers, G2_powers

def encrypted_dot(ec_pts, coeffs): 
    # elliptic curve dot product for tau powers with poly coefficients
    return reduce(add, (multiply(pt, int(c)) for pt, c in zip(ec_pts, coeffs)), Z1)

def t_eval_G1_powers(t, tau, poly_degree):
    # encrypt pre evaluated t at powers of tau
    return [multiply(G1, int(t(tau) * (tau**i))) for i in range(poly_degree)]
        





