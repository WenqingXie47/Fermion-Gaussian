import numpy as np


def random_real_skew_matrix(dim):
    m = np.random.rand(dim,dim)
    real_skew = np.mat((m-m.T)/2)
    return real_skew


def random_hermitian_matrix(dim):
    re = np.random.rand(dim,dim)
    im = np.random.rand(dim,dim)
    matrix = np.mat(re + im*1j)
    hermitian = matrix + matrix.H
    return hermitian