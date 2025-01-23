import numpy as np
from numpy import linalg as LA

def dirac_hamiltonian(A,B):
    pass




class Hamiltonian:

    def __init__(self, matrix):
        matrix = np.mat(matrix)
        assert(Hamiltonian.is_hermitian(matrix))
        self.value = matrix

    @staticmethod
    def is_hermitian(matrix):
        return np.allclose(matrix.H, matrix)

    def to_diagonal(self):
        eigen_energy, U = LA.eig(self.value)
        return np.diag(eigen_energy), U
    

    # correlation matrix is defined by <aa^dagger>
    def to_mixed_state_correlation(self, beta=1.0):
        # ham_diag is an 1D array containing eigenvalues of hamiltonian
        ham_diag, U = LA.eig(self.value) 
        correlation_diag = 1/(1+np.exp(ham_diag*beta))
        correlation = U @ np.diag(correlation_diag) @ U.H
        return correlation


    # correlation matrix C_ij is defined by <c^dagger_i c_j>
    # zero temperature, ground state
    def to_ground_state_correlation(self):
        # ham_diag is an 1D array containing eigenvalues of hamiltonian
        ham_diag, U = LA.eig(self.value)
        # 
        correlation_diag = np.where(ham_diag<0, 1, 0)
        correlation = U @ np.diag(correlation_diag) @ U.H
        return correlation
    






