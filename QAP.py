import numpy as np
import galois
from py_ecc.bn128 import curve_order
from functools import reduce


class QAP:
    """
    The QAP class modifies the R1CS matrices (L, R, O) into a Quadratic Arithmetic Program
    format. This is done via lagrange interpolation of the columns of the R1CS matrices.
    """

    def __init__(self):
        
        # R1CS matrices --- Ls * Rs = Os ,[1, out, x, y, v1inter, v1, v2]
        self.L = np.array([[0,0,1,0,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,4,0,0,0,0],
                           [0,0,0,1,0,0,0]])

        self.R = np.array([[0,0,1,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]])
    
        self.O = np.array([[0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1],
                           [67,0,0,0,0,-1,-1]])
        
        # galois field w/ matching modulus to bn128 
        print("initializing a large field, will take a moment...")
        self.GF = galois.GF(curve_order) 
        self.curve_order = curve_order
        # self.GF, self.curve_order = galois.GF(79), 79
        
    def R1CS_to_QAP(self):
        # interpolate matrices and compute t
        U_poly, V_poly, W_poly = self.poly_interpolate_matrices()
        t = self.compute_t(degree=4)
        return U_poly, V_poly, W_poly, t
    
    def poly_interpolate_matrices(self):

        # convert np arr to galois field arr, handling negatives
        L_galois, R_galois, O_galois = [self.encode_array(arr) for arr in [self.L, self.R, self.O]]

        # r1cs column polynomial interpolation over 1,2,3,4
        interpolate = lambda col: galois.lagrange_poly(self.GF(np.array([1,2,3,4])), col)
        U_poly = np.apply_along_axis(interpolate, 0, L_galois)
        V_poly = np.apply_along_axis(interpolate, 0, R_galois)
        W_poly = np.apply_along_axis(interpolate, 0, O_galois)
        return U_poly, V_poly, W_poly
    
    def encode_array(self, arr: np.array):
        # convert np arr to galois field arr, handling negatives
        return self.GF(np.array(arr, dtype=int) % self.curve_order)
    
    def compute_t(self, degree):  
        # t = (x - 1)(x - 2)(x - 3)(x - 4)
        x_min_i = [galois.Poly([1, self.curve_order - i], field = self.GF) for i in range(1, 5)]
        return reduce(lambda x, y: x * y, x_min_i) # NOTE: poly expansion
    


if __name__ == "__main__":

    qap = QAP()

    # convert R1CS to QAP
    U_poly, V_poly, W_poly, t = qap.R1CS_to_QAP()

    # get galois field 
    GF = qap.GF
    curve_order = qap.curve_order
