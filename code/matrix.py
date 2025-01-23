import numpy as np
import scipy as sp



def omega_matrix(dim):
    id = np.identity(dim, dtype=np.complex128)
    omega = np.block([[id, id], 
                      [id* (1j), id*(-1j)]])
    omega *= (1/np.sqrt(2))
    return omega





def generate_random_real_skew_matrix(dim):
    m = np.random.rand(dim,dim)
    real_skew = (m-m.T)/2
    return real_skew





def diag_real_skew(matrix):

    # verify that the matrix is real skew
    assert (np.allclose(matrix, -matrix.T))
    assert (np.allclose(matrix, matrix.conjugate()))

    # verify the dimension is even
    dim = matrix.shape[0]
    assert(dim%2==0)

    # M is real skew-symmetric, then iM is Hermitian
    eigenvalues, eigenvectors = np.linalg.eig(matrix*1j)

    # sort the eigenvalues in descent order 
    descent_perm = np.argsort(np.real(eigenvalues))[::-1]

    # sort in pairs: e.g. [2,-2,1,-1,0,0]
    perm = np.empty_like(descent_perm)
    perm[::2]  = descent_perm[:dim//2]
    perm[1::2]  = descent_perm[dim//2:][::-1]


    # permute eigenvalues and eigenvectors
    ordered_values = np.real(eigenvalues[perm])
    ordered_vectors = eigenvectors[:,perm]

    # verify that can reverse the transform
    # m = np.real(ordered_vectors @ np.diag(-1j*ordered_values) @ np.mat(ordered_vectors).H)
    # assert(np.allclose(m,matrix))

    # change to real basis
    block = np.array([[1,1],[1j,-1j]]) * (1/np.sqrt(2))
    block_array = [block]*(dim//2)
    transform = sp.linalg.block_diag(*block_array)
    transform = np.mat(transform)


    block_diagonal = np.real(transform @ np.diag(-1j*ordered_values) @ transform.H)
    vectors = np.real(ordered_vectors @ transform.H)

    return block_diagonal, vectors