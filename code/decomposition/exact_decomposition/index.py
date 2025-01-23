import numpy as np
from numpy import linalg as LA


def calculate_index(M, dim_A):
    M_AB = M[:dim_A,dim_A:]
    M_BA = M[dim_A:,:dim_A]

    U_A, Sin_AB, Vh_B =  LA.svd(M_AB)
    U_B, Sin_BA, Vh_A =  LA.svd(M_BA)


    index = np.sum(Sin_BA) - np.sum(Sin_AB)
    return index